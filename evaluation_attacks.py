from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class AttackConfig:
    name: str = "clean"
    malicious_client: str = "A"
    poison_fraction: float = 0.20
    target_label: int = 1
    trigger_feature_index: int = 0
    trigger_value: float = 5.0
    update_scale: float = 3.0


def poison_training_dataset(
    dataset: TensorDataset,
    config: AttackConfig,
    rng: np.random.Generator,
) -> TensorDataset:
    """Return a poisoned copy of one client's training dataset.

    Supported attacks:
      * clean: no change
      * label_flip: flip the labels of a random fraction of records
      * backdoor: set one feature to a trigger value and force a target label

    The survey supplied with the project classifies label-flipping and
    trigger-based poisoning as backdoor/data-poisoning attacks. This function
    implements simple tabular versions suitable for controlled experiments.
    """
    features, labels = dataset.tensors
    x = features.detach().clone()
    y = labels.detach().clone()

    if config.name in {"clean", "sign_flip"}:
        return TensorDataset(x, y)

    if not 0.0 <= config.poison_fraction <= 1.0:
        raise ValueError("poison_fraction must lie in [0, 1]")

    count = int(round(len(y) * config.poison_fraction))
    if count <= 0:
        return TensorDataset(x, y)

    selected = rng.choice(len(y), size=min(count, len(y)), replace=False)
    selected_t = torch.as_tensor(selected, dtype=torch.long)

    if config.name == "label_flip":
        y[selected_t] = 1.0 - y[selected_t]
    elif config.name == "backdoor":
        if not 0 <= config.trigger_feature_index < x.shape[1]:
            raise ValueError("trigger_feature_index is outside the feature vector")
        x[selected_t, config.trigger_feature_index] = float(config.trigger_value)
        y[selected_t] = float(config.target_label)
    else:
        raise ValueError(f"Unsupported data attack: {config.name}")

    return TensorDataset(x, y)


def poison_update(
    update: np.ndarray,
    config: AttackConfig,
    max_abs_value: Optional[float] = None,
) -> np.ndarray:
    """Apply a bounded Byzantine sign-flip/model-poisoning attack.

    The same transformed update is used for ordinary FL and Hybrid PET so the
    integrity comparison is fair. Hybrid PET encrypts the malicious update but
    does not authenticate whether the local training procedure was honest.
    """
    result = np.asarray(update, dtype=np.float64).copy()
    if config.name != "sign_flip":
        return result

    result = -float(config.update_scale) * result
    if max_abs_value is not None:
        result = np.clip(result, -max_abs_value, max_abs_value)
    return result


def make_triggered_dataset(
    dataset: TensorDataset,
    feature_index: int,
    trigger_value: float,
    target_label: int,
) -> TensorDataset:
    """Create a triggered test set for measuring backdoor attack success."""
    features, labels = dataset.tensors
    x = features.detach().clone()
    y = labels.detach().clone()
    if not 0 <= feature_index < x.shape[1]:
        raise ValueError("feature_index is outside the feature vector")
    x[:, feature_index] = float(trigger_value)
    y[:] = float(target_label)
    return TensorDataset(x, y)
