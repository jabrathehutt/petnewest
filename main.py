from __future__ import annotations

import numpy as np
import torch
import flwr as fl
from flwr.common import Context

import config as cfg
from audit import initialize_dimensions, run_crypto_self_test
from bgv import ManualBGV, ThresholdDecryptor
from dkg import DistributedBGVSetup
from flower_client import DatasetMetadata, SecureSmokingClient
from flower_server import ThesisProtocolStrategy
from ring import Ring
from simd import SIMDEncoder


def print_summary(strategy: ThesisProtocolStrategy) -> None:
    model_dimension, ring_degree, row_size = (
        cfg.RUNTIME.require_dimensions()
    )

    print("\n" + "=" * 78)
    print("PROTOCOL SUMMARY")
    print("=" * 78)
    print(f"Five FL clients: {cfg.CLIENT_NAMES}")
    print(
        f"Threshold access structure: "
        f"{cfg.THRESHOLD}-of-{cfg.CLIENT_COUNT}"
    )
    print(f"Model dimension: {model_dimension}")
    print(f"Ring degree N: {ring_degree}")
    print(f"First-row SIMD capacity: {row_size}")
    print(f"Plaintext modulus p: {cfg.P_MODULUS}")
    print(f"Ciphertext modulus q: {cfg.Q_MODULUS}")
    print(
        "Public per-client quantized slot bound: "
        f"{cfg.PUBLIC_CLIENT_SLOT_BOUND}"
    )
    print("Collective secret reconstructed during protocol: no")
    print("Relinearization key: canonical Enc_s(B^j*s^2)")
    print(
        "Rotation keys: canonical "
        "Enc_s(B^j*sigma_k(s))"
    )
    print("Balanced coefficient decomposition: enabled")
    print("Runtime public noise tracking: enabled")
    print("Inner product evaluated in every communication round")

    if strategy.round_history:
        latest_round = strategy.round_history[-1]

        print(
            f"Final authorized scalar: "
            f"{latest_round['authorized_scalar']:.8f}"
        )
        print(
            f"Final plaintext validation: "
            f"{latest_round['validation_scalar']:.8f}"
        )

        # Optional: print a stored modulus report only when one exists.
        latest_bound = latest_round.get(
            "protocol_modulus_bound"
        )

        if latest_bound is not None:
            print(
                "Final public query-specific minimum p: "
                f"{latest_bound.get('minimum_p_query_specific_public', 'unavailable')}"
            )
            print(
                "Final raw-representative symbolic minimum q: "
                f"{latest_bound.get('minimum_q_public_runtime', 'unavailable')}"
            )
            print(
                "Final exact-execution minimum q: "
                f"{latest_bound.get('minimum_q_observed', 'unavailable')}"
            )

    if strategy.evaluation_history:
        latest = strategy.evaluation_history[-1]
        print(
            f"Final distributed test loss: "
            f"{latest['loss']:.6f}"
        )
        print(
            f"Final distributed test accuracy: "
            f"{latest['accuracy']:.4f}"
        )


def main() -> None:
    np.random.seed(cfg.RANDOM_SEED)
    torch.manual_seed(cfg.RANDOM_SEED)
    torch.set_num_threads(1)

    metadata = DatasetMetadata.load(cfg.METADATA_FILE)
    initialize_dimensions(metadata)
    _, ring_degree, _ = cfg.RUNTIME.require_dimensions()

    ring_q = Ring(ring_degree, cfg.Q_MODULUS)
    ring_p = Ring(ring_degree, cfg.P_MODULUS)
    encoder = SIMDEncoder(ring_p)

    print("\n" + "=" * 78)
    print("TEXTBOOK RING-BGV + 3-OF-5 LISS + FLOWER")
    print("=" * 78)

    setup = DistributedBGVSetup(
        ring_q=ring_q,
        encoder=encoder,
        participant_ids=cfg.CLIENT_IDS,
        threshold=cfg.THRESHOLD,
        seed=cfg.RANDOM_SEED,
    )
    setup_result = setup.generate()

    bgv = ManualBGV(
        ring_q=ring_q,
        ring_p=ring_p,
        encoder=encoder,
        public_key=setup_result.public_key,
        evaluation_keys=setup_result.evaluation_keys,
        seed=cfg.RANDOM_SEED + 100,
    )
    decryptor = ThresholdDecryptor(
        ring_q=ring_q,
        ring_p=ring_p,
        encoder=encoder,
        liss=setup_result.liss,
        shares=setup_result.liss_shares,
        seed=cfg.RANDOM_SEED + 200,
    )

    run_crypto_self_test(
        ring_q,
        ring_p,
        encoder,
        bgv,
        decryptor,
        setup_result.local_secret_contributions,
        setup_result.square_shares,
    )

    if cfg.RUNTIME.authorized_query is None:
        raise RuntimeError("Authorized query was not initialized")
    encrypted_query = bgv.encrypt_slots(
        cfg.RUNTIME.authorized_query,
        label="authorized_query",
    )

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        client_id = partition_id + 1
        client_name = cfg.CLIENT_NAMES[partition_id]

        client_bgv = ManualBGV(
            ring_q=ring_q,
            ring_p=ring_p,
            encoder=encoder,
            public_key=setup_result.public_key,
            evaluation_keys=setup_result.evaluation_keys,
            seed=cfg.RANDOM_SEED + 1_000 + client_id,
        )
        return SecureSmokingClient(
            client_id=client_id,
            client_name=client_name,
            metadata=metadata,
            bgv=client_bgv,
        ).to_client()

    strategy = ThesisProtocolStrategy(
        ring_q=ring_q,
        bgv=bgv,
        decryptor=decryptor,
        encrypted_query=encrypted_query,
        committee_ids=cfg.CLIENT_IDS,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.CLIENT_COUNT,
        config=fl.server.ServerConfig(num_rounds=cfg.NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 0.5, "num_gpus": 0.0},
    )
    print_summary(strategy)


if __name__ == "__main__":
    main()
