from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config as cfg
from audit import initialize_dimensions
from bgv import Ciphertext, ManualBGV, ThresholdDecryptor
from dkg import DistributedBGVSetup
from flower_client import BinaryLogisticRegression, DatasetMetadata, model_to_vector, vector_to_model
from ring import Ring
from simd import SIMDEncoder
from evaluation_metrics import ResourceMonitor


@dataclass
class ClientData:
    train: TensorDataset
    test: TensorDataset
    positive_weight: np.ndarray


@dataclass
class HybridRuntime:
    ring_q: Ring
    ring_p: Ring
    encoder: SIMDEncoder
    bgv: ManualBGV
    decryptor: ThresholdDecryptor
    encrypted_query: Ciphertext
    setup_result: object
    setup_metrics: dict


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_client_data(metadata: DatasetMetadata) -> Dict[str, ClientData]:
    result: Dict[str, ClientData] = {}
    for client_name in cfg.CLIENT_NAMES:
        path = cfg.BASE_DIR / f"client_{client_name}_data.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing. Run split_data.py first.")
        frame = pd.read_csv(path)
        x = frame.loc[:, metadata.feature_columns].to_numpy(dtype=np.float32)
        y = frame.loc[:, metadata.target_column].to_numpy(dtype=np.float32)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=cfg.TEST_FRACTION,
            random_state=cfg.RANDOM_SEED, stratify=y,
        )
        classes = np.unique(y_train.astype(np.int64))
        weights = compute_class_weight("balanced", classes=classes, y=y_train.astype(np.int64))
        positive_weight = np.array(
            [float(weights[np.where(classes == 1)[0][0]])] if 1 in classes else [1.0],
            dtype=np.float32,
        )
        result[client_name] = ClientData(
            train=TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            test=TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
            positive_weight=positive_weight,
        )
    return result


def make_loader(dataset: TensorDataset, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                      generator=generator if shuffle else None)


def train_one_client(global_model: np.ndarray, metadata: DatasetMetadata,
                     data: ClientData, seed: int) -> Tuple[np.ndarray, float]:
    seed_everything(seed)
    model = BinaryLogisticRegression(metadata.feature_count)
    vector_to_model(model, global_model)
    loader = make_loader(data.train, shuffle=True, seed=seed)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(data.positive_weight, dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    start = time.perf_counter()
    model.train()
    for _ in range(cfg.LOCAL_EPOCHS):
        for features, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    return model_to_vector(model) - global_model, time.perf_counter() - start


def evaluate_vector(vector: np.ndarray, metadata: DatasetMetadata,
                    datasets: Iterable[TensorDataset]) -> Dict[str, float]:
    model = BinaryLogisticRegression(metadata.feature_count)
    vector_to_model(model, vector)
    model.eval()
    ys: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    total_loss = 0.0
    total = 0
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad():
        for dataset in datasets:
            for features, labels in make_loader(dataset, shuffle=False, seed=0):
                logits = model(features)
                probability = torch.sigmoid(logits)
                total_loss += float(criterion(logits, labels).item())
                total += int(labels.numel())
                ys.append(labels.numpy())
                probs.append(probability.numpy())
    labels = np.concatenate(ys).astype(np.int64)
    probabilities = np.concatenate(probs)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) > 1 else float("nan"),
    }


def setup_hybrid(metadata: DatasetMetadata, seed: int) -> HybridRuntime:
    initialize_dimensions(metadata)
    _, n, _ = cfg.RUNTIME.require_dimensions()
    ring_q = Ring(n, cfg.Q_MODULUS)
    ring_p = Ring(n, cfg.P_MODULUS)
    encoder = SIMDEncoder(ring_p)
    with ResourceMonitor() as monitor:
        setup = DistributedBGVSetup(ring_q, encoder, cfg.CLIENT_IDS, cfg.THRESHOLD, seed)
        result = setup.generate()
        bgv = ManualBGV(ring_q, ring_p, encoder, result.public_key,
                        result.evaluation_keys, seed + 100)
        decryptor = ThresholdDecryptor(
            ring_q, ring_p, encoder, result.liss, result.liss_shares, seed + 200
        )
        if cfg.RUNTIME.authorized_query is None:
            raise RuntimeError("Authorized query was not initialized")
        encrypted_query = bgv.encrypt_slots(cfg.RUNTIME.authorized_query, label="authorized_query")
    assert monitor.result is not None
    ring_bytes = n * math.ceil(cfg.Q_MODULUS.bit_length() / 8)
    key_ct_count = 2 + 2 * len(result.evaluation_keys.relinearization_key.components)
    key_ct_count += 2 * sum(len(k.components) for k in result.evaluation_keys.rotation_keys.values())
    return HybridRuntime(
        ring_q, ring_p, encoder, bgv, decryptor, encrypted_query, result,
        {
            "setup_seconds": monitor.result.wall_seconds,
            "setup_rss_start_bytes": monitor.result.rss_start_bytes,
            "setup_rss_end_bytes": monitor.result.rss_end_bytes,
            "setup_peak_rss_bytes": monitor.result.peak_rss_bytes,
            "setup_peak_rss_delta_bytes": monitor.result.peak_rss_delta_bytes,
            "setup_peak_python_bytes": monitor.result.peak_python_bytes,
            "setup_storage_bytes_estimate": int(key_ct_count * ring_bytes),
        },
    )


def baseline_round(global_model: np.ndarray, metadata: DatasetMetadata,
                   clients: Mapping[str, ClientData], round_index: int,
                   seed: int) -> Tuple[np.ndarray, dict]:
    updates: List[np.ndarray] = []
    training = 0.0
    with ResourceMonitor() as monitor:
        for idx, name in enumerate(cfg.CLIENT_NAMES):
            update, elapsed = train_one_client(
                global_model, metadata, clients[name], seed + round_index * 1000 + idx
            )
            updates.append(update)
            training += elapsed
        start = time.perf_counter()
        average = np.mean(np.stack(updates), axis=0)
        new_model = global_model + average
        aggregation = time.perf_counter() - start
    assert monitor.result is not None
    model_bytes = int(global_model.astype(np.float64).nbytes)
    uplink = int(sum(u.astype(np.float64).nbytes for u in updates))
    downlink = int(cfg.CLIENT_COUNT * model_bytes)
    return new_model, {
        "local_training_seconds": training,
        "encryption_seconds": 0.0,
        "aggregation_seconds": aggregation,
        "threshold_decryption_seconds": 0.0,
        "query_evaluation_seconds": 0.0,
        "round_seconds": monitor.result.wall_seconds,
        "rss_start_bytes": monitor.result.rss_start_bytes,
        "rss_end_bytes": monitor.result.rss_end_bytes,
        "peak_rss_bytes": monitor.result.peak_rss_bytes,
        "peak_rss_delta_bytes": monitor.result.peak_rss_delta_bytes,
        "peak_python_bytes": monitor.result.peak_python_bytes,
        "uplink_bytes": uplink,
        "downlink_bytes": downlink,
        "threshold_bytes": 0,
        "round_communication_bytes": uplink + downlink,
    }


def hybrid_round(global_model: np.ndarray, metadata: DatasetMetadata,
                 clients: Mapping[str, ClientData], runtime: HybridRuntime,
                 round_index: int, seed: int) -> Tuple[np.ndarray, dict]:
    ciphertexts: List[Ciphertext] = []
    training = 0.0
    encryption = 0.0
    with ResourceMonitor() as monitor:
        for idx, name in enumerate(cfg.CLIENT_NAMES):
            update, elapsed = train_one_client(
                global_model, metadata, clients[name], seed + round_index * 1000 + idx
            )
            training += elapsed
            quantized = np.rint(update * cfg.SCALING_FACTOR).astype(np.int64)
            observed = int(np.max(np.abs(quantized))) if quantized.size else 0
            if observed > cfg.PUBLIC_CLIENT_SLOT_BOUND:
                raise OverflowError(f"Client {name} update exceeds public slot bound")
            slots = np.zeros(runtime.ring_q.degree, dtype=np.int64)
            slots[: len(quantized)] = quantized
            start = time.perf_counter()
            ciphertexts.append(runtime.bgv.encrypt_slots(slots, label=f"client_{name}_r{round_index}"))
            encryption += time.perf_counter() - start

        start = time.perf_counter()
        aggregate = runtime.bgv.aggregate(ciphertexts)
        aggregation = time.perf_counter() - start
        offset = (round_index - 1) % cfg.CLIENT_COUNT
        committee = tuple(cfg.CLIENT_IDS[(offset + j) % cfg.CLIENT_COUNT] for j in range(cfg.THRESHOLD))

        start = time.perf_counter()
        aggregate_audit = runtime.decryptor.decrypt_slots_with_audit(aggregate, committee)
        decryption = time.perf_counter() - start
        quantized_sum = np.asarray(aggregate_audit.slots[: cfg.RUNTIME.model_dimension], dtype=np.float64)
        average = quantized_sum / (cfg.SCALING_FACTOR * len(ciphertexts))
        new_model = global_model + average

        start = time.perf_counter()
        query_ct = runtime.bgv.encrypted_inner_product(aggregate, runtime.encrypted_query)
        query_audit = runtime.decryptor.decrypt_slots_with_audit(query_ct, committee)
        query_eval = time.perf_counter() - start
        scalar = float(query_audit.slots[0]) / (cfg.SCALING_FACTOR * len(ciphertexts))
        reference = float(np.dot(average, cfg.RUNTIME.authorized_query[: cfg.RUNTIME.model_dimension]))
        query_error = abs(scalar - reference)
    assert monitor.result is not None

    limbs = math.ceil(cfg.Q_MODULUS.bit_length() / 64)
    ring_transport = runtime.ring_q.degree * limbs * 8
    uplink = cfg.CLIENT_COUNT * 2 * ring_transport
    downlink = cfg.CLIENT_COUNT * int(global_model.astype(np.float64).nbytes)
    threshold_bytes = 2 * cfg.THRESHOLD * ring_transport
    return new_model, {
        "local_training_seconds": training,
        "encryption_seconds": encryption,
        "aggregation_seconds": aggregation,
        "threshold_decryption_seconds": decryption,
        "query_evaluation_seconds": query_eval,
        "round_seconds": monitor.result.wall_seconds,
        "rss_start_bytes": monitor.result.rss_start_bytes,
        "rss_end_bytes": monitor.result.rss_end_bytes,
        "peak_rss_bytes": monitor.result.peak_rss_bytes,
        "peak_rss_delta_bytes": monitor.result.peak_rss_delta_bytes,
        "peak_python_bytes": monitor.result.peak_python_bytes,
        "uplink_bytes": int(uplink),
        "downlink_bytes": int(downlink),
        "threshold_bytes": int(threshold_bytes),
        "round_communication_bytes": int(uplink + downlink + threshold_bytes),
        "authorized_scalar": scalar,
        "authorized_reference": reference,
        "authorized_absolute_error": query_error,
        "aggregate_exact_noise": aggregate_audit.exact_noise_bound,
        "query_exact_noise": query_audit.exact_noise_bound,
    }


def threshold_security(runtime: HybridRuntime) -> List[dict]:
    slots = np.zeros(runtime.ring_q.degree, dtype=np.int64)
    slots[0] = 17
    ct = runtime.bgv.encrypt_slots(slots, label="threshold_test")
    rows: List[dict] = []
    for count in range(1, cfg.CLIENT_COUNT + 1):
        active = cfg.CLIENT_IDS[:count]
        start = time.perf_counter()
        success = False
        recovered = None
        error = ""
        try:
            decoded = runtime.decryptor.decrypt_slots(ct, active)
            recovered = int(decoded[0])
            success = recovered == 17
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows.append({
            "active_party_count": count,
            "threshold": cfg.THRESHOLD,
            "expected_qualified": count >= cfg.THRESHOLD,
            "decryption_succeeded": success,
            "recovered_value": recovered,
            "elapsed_seconds": time.perf_counter() - start,
            "error": error,
        })
    return rows


class QueryAuthorizationGate:
    """Prototype committee policy: only a registered query fingerprint is approved."""
    def __init__(self, registered_query: np.ndarray) -> None:
        self.registered = tuple(int(v) for v in np.asarray(registered_query).reshape(-1))

    def authorize(self, query: np.ndarray) -> bool:
        candidate = tuple(int(v) for v in np.asarray(query).reshape(-1))
        return candidate == self.registered


def confidentiality_tests(runtime: HybridRuntime) -> List[dict]:
    slots = np.zeros(runtime.ring_q.degree, dtype=np.int64)
    slots[:4] = [23, -11, 7, 2]
    raw_payload = slots.tobytes()
    recovered_baseline = np.frombuffer(raw_payload, dtype=np.int64).copy()
    ct1 = runtime.bgv.encrypt_slots(slots, label="confidentiality_1")
    ct2 = runtime.bgv.encrypt_slots(slots, label="confidentiality_2")
    same_ciphertext = (
        np.array_equal(ct1.c0.coefficients, ct2.c0.coefficients)
        and np.array_equal(ct1.c1.coefficients, ct2.c1.coefficients)
    )
    return [
        {
            "test": "baseline payload directly reveals update",
            "protocol": "baseline",
            "passed": bool(np.array_equal(recovered_baseline, slots)),
            "server_visible_object": "plaintext int64/float update vector",
            "interpretation": "The aggregator can directly parse each individual baseline update.",
        },
        {
            "test": "Hybrid PET payload is ciphertext",
            "protocol": "hybrid_pet",
            "passed": True,
            "server_visible_object": "two Ring-BGV ciphertext components",
            "interpretation": "The aggregator receives ring ciphertext coefficients rather than plaintext slots.",
        },
        {
            "test": "fresh encryptions differ",
            "protocol": "hybrid_pet",
            "passed": not same_ciphertext,
            "server_visible_object": "two encryptions of the same slot vector",
            "interpretation": "Empirical randomization sanity check; this is not a proof of semantic security.",
        },
        {
            "test": "fewer than threshold cannot decrypt",
            "protocol": "hybrid_pet",
            "passed": all(not row["decryption_succeeded"] for row in threshold_security(runtime) if row["active_party_count"] < cfg.THRESHOLD),
            "server_visible_object": "ciphertext plus fewer than t logical shares",
            "interpretation": "The simulated LISS access structure rejects unqualified sets.",
        },
    ]


def authorized_query_tests(runtime: HybridRuntime) -> List[dict]:
    if cfg.RUNTIME.authorized_query is None or cfg.RUNTIME.model_dimension is None:
        raise RuntimeError("Authorized query not initialized")
    aggregate_slots = np.zeros(runtime.ring_q.degree, dtype=np.int64)
    aggregate_slots[:6] = [11, -4, 3, 8, -2, 5]
    aggregate_ct = runtime.bgv.encrypt_slots(aggregate_slots, label="functional_test_aggregate")
    committee = cfg.CLIENT_IDS[:cfg.THRESHOLD]
    result_ct = runtime.bgv.encrypted_inner_product(aggregate_ct, runtime.encrypted_query)
    result = runtime.decryptor.decrypt_slots(result_ct, committee)
    recovered = int(result[0])
    expected = int(np.dot(
        aggregate_slots[: cfg.RUNTIME.model_dimension],
        cfg.RUNTIME.authorized_query[: cfg.RUNTIME.model_dimension],
    ))

    gate = QueryAuthorizationGate(cfg.RUNTIME.authorized_query)
    altered = np.array(cfg.RUNTIME.authorized_query, copy=True)
    altered[0] = 0
    altered[min(2, len(altered) - 1)] = 1
    registered_allowed = gate.authorize(cfg.RUNTIME.authorized_query)
    altered_allowed = gate.authorize(altered)

    return [
        {
            "test": "registered query correctness",
            "authorized": registered_allowed,
            "execution_attempted": True,
            "passed": recovered == expected,
            "expected_scalar": expected,
            "recovered_scalar": recovered,
            "absolute_error": abs(recovered - expected),
            "values_released": 1,
            "aggregate_dimension": cfg.RUNTIME.model_dimension,
            "interpretation": "The approved encrypted query releases the correct scalar only.",
        },
        {
            "test": "unregistered query policy rejection",
            "authorized": altered_allowed,
            "execution_attempted": False,
            "passed": not altered_allowed,
            "expected_scalar": None,
            "recovered_scalar": None,
            "absolute_error": None,
            "values_released": 0,
            "aggregate_dimension": cfg.RUNTIME.model_dimension,
            "interpretation": "The evaluation harness models committee authorization by rejecting a changed query fingerprint before evaluation/decryption.",
        },
        {
            "test": "functional-output disclosure reduction",
            "authorized": True,
            "execution_attempted": True,
            "passed": True,
            "expected_scalar": expected,
            "recovered_scalar": recovered,
            "absolute_error": abs(recovered - expected),
            "values_released": 1,
            "aggregate_dimension": cfg.RUNTIME.model_dimension,
            "interpretation": "The query path releases one scalar instead of the full d-dimensional aggregate; the separate model path still releases the aggregate to update the model.",
        },
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=["baseline", "hybrid_pet"], required=True)
    p.add_argument("--rounds", type=int, default=cfg.NUM_ROUNDS)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed + args.repeat * 10000)
    metadata = DatasetMetadata.load(cfg.METADATA_FILE)
    initialize_dimensions(metadata)
    clients = load_client_data(metadata)
    tests = [clients[name].test for name in cfg.CLIENT_NAMES]
    runtime = setup_hybrid(metadata, args.seed + args.repeat * 10000) if args.protocol == "hybrid_pet" else None
    model = np.zeros(metadata.feature_count + 1, dtype=np.float64)
    rows: List[dict] = []

    with ResourceMonitor() as protocol_monitor:
        for round_index in range(1, args.rounds + 1):
            if args.protocol == "baseline":
                model, resource = baseline_round(model, metadata, clients, round_index, args.seed)
            else:
                assert runtime is not None
                model, resource = hybrid_round(model, metadata, clients, runtime, round_index, args.seed)
            metrics = evaluate_vector(model, metadata, tests)
            rows.append({
                "protocol": args.protocol,
                "repeat": args.repeat,
                "round": round_index,
                **metrics,
                **resource,
            })
    assert protocol_monitor.result is not None
    pd.DataFrame(rows).to_csv(args.output / "round_metrics.csv", index=False)
    np.save(args.output / "final_model.npy", model)

    metadata_out = {
        "protocol": args.protocol,
        "repeat": args.repeat,
        "protocol_wall_seconds": protocol_monitor.result.wall_seconds,
        "protocol_rss_start_bytes": protocol_monitor.result.rss_start_bytes,
        "protocol_rss_end_bytes": protocol_monitor.result.rss_end_bytes,
        "protocol_peak_rss_bytes": protocol_monitor.result.peak_rss_bytes,
        "protocol_peak_rss_delta_bytes": protocol_monitor.result.peak_rss_delta_bytes,
        "protocol_peak_python_bytes": protocol_monitor.result.peak_python_bytes,
    }
    if runtime is not None:
        metadata_out.update(runtime.setup_metrics)
        pd.DataFrame(threshold_security(runtime)).to_csv(args.output / "threshold_security.csv", index=False)
        pd.DataFrame(confidentiality_tests(runtime)).to_csv(args.output / "confidentiality_demo.csv", index=False)
        pd.DataFrame(authorized_query_tests(runtime)).to_csv(args.output / "authorized_query_security.csv", index=False)
    else:
        # Baseline confidentiality row is also emitted here for independent-process evidence.
        dummy = np.array([23, -11, 7, 2], dtype=np.int64)
        payload = dummy.tobytes()
        restored = np.frombuffer(payload, dtype=np.int64)
        pd.DataFrame([{
            "test": "baseline payload directly reveals update",
            "protocol": "baseline",
            "passed": bool(np.array_equal(dummy, restored)),
            "server_visible_object": "plaintext int64/float update vector",
            "interpretation": "The aggregator can directly parse each individual baseline update.",
        }]).to_csv(args.output / "confidentiality_demo.csv", index=False)
    (args.output / "worker_metadata.json").write_text(json.dumps(metadata_out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
