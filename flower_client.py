from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import Parameters, parameters_to_ndarrays

import config as cfg
from bgv import ManualBGV
from transport import encode_bigint_coefficients

@dataclass(frozen=True)
class DatasetMetadata:
    target_column: str
    feature_columns: Tuple[str, ...]
    feature_count: int

    @classmethod
    def load(cls, path: Path) -> "DatasetMetadata":
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} was not found. Run "
                "`python3 split_data.py smoking.csv` first."
            )
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            target_column=str(raw["target_column"]),
            feature_columns=tuple(raw["feature_columns"]),
            feature_count=int(raw["feature_count"]),
        )


class BinaryLogisticRegression(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_count, 1, bias=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def model_to_vector(model: nn.Module) -> np.ndarray:
    values = torch.cat(
        [
            parameter.detach().cpu().reshape(-1)
            for parameter in model.parameters()
        ]
    )
    return values.numpy().astype(np.float64, copy=True)


def vector_to_model(
    model: nn.Module,
    vector: np.ndarray,
) -> None:
    values = np.asarray(vector, dtype=np.float32)
    expected = sum(parameter.numel() for parameter in model.parameters())
    if values.shape != (expected,):
        raise ValueError(
            f"Expected model vector {(expected,)}, got {values.shape}"
        )

    tensor = torch.from_numpy(values)
    cursor = 0

    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(
                tensor[cursor : cursor + count].reshape_as(parameter)
            )
            cursor += count


def load_client_partition(
    client_name: str,
    metadata: DatasetMetadata,
) -> tuple[TensorDataset, TensorDataset, np.ndarray]:
    path = cfg.BASE_DIR / f"client_{client_name}_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} is missing. Run split_data.py first."
        )

    frame = pd.read_csv(path)
    features = frame.loc[
        :,
        metadata.feature_columns,
    ].to_numpy(dtype=np.float32)
    labels = frame.loc[
        :,
        metadata.target_column,
    ].to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.TEST_FRACTION,
        random_state=cfg.RANDOM_SEED,
        stratify=labels,
    )

    classes = np.unique(y_train.astype(np.int64))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(np.int64),
    )
    positive_weight = np.array(
        [
            float(
                class_weights[
                    np.where(classes == 1)[0][0]
                ]
            )
        ]
        if 1 in classes
        else [1.0],
        dtype=np.float32,
    )

    return (
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(y_train),
        ),
        TensorDataset(
            torch.from_numpy(x_test),
            torch.from_numpy(y_test),
        ),
        positive_weight,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            total_loss += float(criterion(logits, labels).item())
            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).to(labels.dtype)
            total_correct += int(
                (predictions == labels).sum().item()
            )
            total_examples += int(labels.numel())

    if total_examples == 0:
        raise ValueError("Empty dataset")

    return (
        total_loss / total_examples,
        total_correct / total_examples,
    )


def train_local_model(
    model: nn.Module,
    loader: DataLoader,
    positive_weight: np.ndarray,
) -> tuple[float, float]:
    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight,
            dtype=torch.float32,
        )
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    for _ in range(cfg.LOCAL_EPOCHS):
        for features, labels in loader:
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    return evaluate_model(model, loader)


def quantize_and_pack(update: np.ndarray) -> np.ndarray:
    if cfg.RUNTIME.model_dimension is None or cfg.RUNTIME.ring_degree is None:
        raise RuntimeError("Dimensions are not initialized")

    quantized = np.rint(
        np.asarray(update, dtype=np.float64)
        * cfg.SCALING_FACTOR
    ).astype(np.int64)

    if quantized.shape != (cfg.RUNTIME.model_dimension,):
        raise ValueError("Unexpected update dimension")

    observed_bound = int(np.max(np.abs(quantized))) if quantized.size else 0
    if observed_bound > cfg.PUBLIC_CLIENT_SLOT_BOUND:
        raise OverflowError(
            "Quantized client update violates the public protocol bound: "
            f"observed={observed_bound}, "
            f"allowed={cfg.PUBLIC_CLIENT_SLOT_BOUND}. "
            "Increase the declared bound and re-check p, or introduce an "
            "explicit clipping rule in the protocol design."
        )

    slots = np.zeros(cfg.RUNTIME.ring_degree, dtype=np.int64)
    slots[:cfg.RUNTIME.model_dimension] = quantized
    return slots


# =============================================================================
# FLOWER CLIENT
# =============================================================================


class SecureSmokingClient(fl.client.NumPyClient):
    def __init__(
        self,
        client_id: int,
        client_name: str,
        metadata: DatasetMetadata,
        bgv: ManualBGV,
    ):
        self.client_id = client_id
        self.client_name = client_name
        self.metadata = metadata
        self.bgv = bgv

        # Ray clients run in separate processes, so mutable values assigned to
        # cfg.RUNTIME in main.py are not inherited by the client workers.
        cfg.RUNTIME.model_dimension = metadata.feature_count + 1
        cfg.RUNTIME.ring_degree = bgv.ring_q.degree
        cfg.RUNTIME.row_size = bgv.ring_q.degree // 2

        print(
            f"[Client {client_name} dimension audit] "
            f"features={metadata.feature_count}, "
            f"model_dimension={cfg.RUNTIME.model_dimension}, "
            f"ring_degree={cfg.RUNTIME.ring_degree}"
        )

        train_dataset, test_dataset, positive_weight = load_client_partition(
            client_name,
            metadata,
        )
        self.positive_weight = positive_weight

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            generator=torch.Generator().manual_seed(
                cfg.RANDOM_SEED + client_id
            ),
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
        )
        self.train_examples = len(train_dataset)
        self.test_examples = len(test_dataset)

    @staticmethod
    def read_global_model(parameters) -> np.ndarray:
        arrays = (
            parameters
            if isinstance(parameters, list)
            else parameters_to_ndarrays(parameters)
            if isinstance(parameters, Parameters)
            else []
        )

        if cfg.RUNTIME.model_dimension is None:
            raise RuntimeError(
                "cfg.RUNTIME.model_dimension is not initialized "
                "inside the Ray client worker"
            )

        if not arrays:
            return np.zeros(
                cfg.RUNTIME.model_dimension,
                dtype=np.float64,
            )

        vector = np.asarray(
            arrays[0],
            dtype=np.float64,
        ).reshape(-1)

        expected_shape = (cfg.RUNTIME.model_dimension,)

        if vector.shape != expected_shape:
            raise ValueError(
                "Unexpected global model shape: "
                f"received={vector.shape}, "
                f"expected={expected_shape}"
            )

        return vector

    def get_parameters(self, config):
        if cfg.RUNTIME.model_dimension is None:
            raise RuntimeError("cfg.RUNTIME.model_dimension is not initialized")
        return [
            np.zeros(
                cfg.RUNTIME.model_dimension,
                dtype=np.float64,
            )
        ]

    def fit(self, parameters, config):
        global_model = self.read_global_model(parameters)

        local_model = BinaryLogisticRegression(
            self.metadata.feature_count
        )
        vector_to_model(local_model, global_model)

        loss_before, accuracy_before = evaluate_model(
            local_model,
            self.train_loader,
        )
        loss_after, accuracy_after = train_local_model(
            local_model,
            self.train_loader,
            self.positive_weight,
        )

        local_vector = model_to_vector(local_model)
        update = local_vector - global_model
        packed_update = quantize_and_pack(update)
        ciphertext = self.bgv.encrypt_slots(
            packed_update,
            label=f"client_{self.client_name}_update",
        )

        print(
            f"[Client {self.client_name}] "
            f"loss={loss_before:.4f}->{loss_after:.4f}, "
            f"accuracy={accuracy_before:.4f}->{accuracy_after:.4f}"
        )

        return (
            [
                encode_bigint_coefficients(
                    ciphertext.c0.coefficients,
                    modulus=cfg.Q_MODULUS,
                ),
                encode_bigint_coefficients(
                    ciphertext.c1.coefficients,
                    modulus=cfg.Q_MODULUS,
                ),
            ],
            self.train_examples,
            {
                "client_id": self.client_id,
                "client_name": self.client_name,
                "message_bound": int(ciphertext.message_bound),
                "noise_bound": int(ciphertext.noise_bound),
            },
        )

    def evaluate(self, parameters, config):
        global_model = self.read_global_model(parameters)

        model = BinaryLogisticRegression(
            self.metadata.feature_count
        )
        vector_to_model(model, global_model)

        loss, accuracy = evaluate_model(
            model,
            self.test_loader,
        )
        return (
            float(loss),
            self.test_examples,
            {"accuracy": float(accuracy)},
        )


# =============================================================================
# FLOWER SERVER STRATEGY
# =============================================================================


