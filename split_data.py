
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CLIENT_NAMES = ("A", "B", "C", "D", "E")
RANDOM_SEED = 42
DEFAULT_TARGET_CANDIDATES = (
    "smoking",
    "smoker",
    "target",
    "label",
    "class",
)


def find_target_column(frame: pd.DataFrame) -> str:
    lowered = {str(column).strip().lower(): str(column) for column in frame.columns}

    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]

    # The common Kaggle body-signal smoking dataset places "smoking" last.
    # Falling back to the last column keeps the script usable for close variants.
    return str(frame.columns[-1])


def normalize_binary_target(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("The target column contains missing values")

    if pd.api.types.is_numeric_dtype(series):
        unique = sorted(pd.unique(series))
        if len(unique) != 2:
            raise ValueError(
                f"Expected a binary target, found values: {unique[:10]}"
            )
        mapping = {unique[0]: 0, unique[1]: 1}
        return series.map(mapping).astype(np.int64)

    normalized = series.astype(str).str.strip().str.lower()
    unique = sorted(normalized.unique())
    if len(unique) != 2:
        raise ValueError(
            f"Expected a binary target, found values: {unique[:10]}"
        )

    positive_tokens = {"1", "yes", "true", "smoker", "smoking", "positive"}
    positive = next((value for value in unique if value in positive_tokens), unique[-1])
    return (normalized == positive).astype(np.int64)


def preprocess_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()

    # Remove export artifacts and obvious row identifiers.
    removable = [
        column
        for column in features.columns
        if str(column).lower().startswith("unnamed:")
        or str(column).strip().lower() in {"id", "index", "row_id"}
    ]
    if removable:
        features = features.drop(columns=removable)

    # Replace infinities before imputation.
    features = features.replace([np.inf, -np.inf], np.nan)

    numeric_columns = list(
        features.select_dtypes(include=[np.number, "bool"]).columns
    )
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    for column in numeric_columns:
        median = features[column].median()
        features[column] = features[column].fillna(median)

    for column in categorical_columns:
        mode = features[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        features[column] = features[column].fillna(fill_value).astype(str)

    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        drop_first=False,
        dtype=np.float64,
    )

    if features.empty:
        raise ValueError("No usable feature columns remained after preprocessing")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.to_numpy(dtype=np.float64))

    return pd.DataFrame(
        scaled,
        columns=[str(column) for column in features.columns],
        index=features.index,
    )


def split_kaggle_dataset(csv_path: str) -> None:
    source = Path(csv_path)
    print(f"[*] Locating dataset target file: {source}...")

    if not source.exists():
        raise FileNotFoundError(
            f"Could not find '{source}'. Put smoking.csv in this directory "
            "or pass its path explicitly."
        )

    frame = pd.read_csv(source)
    print(f"[+] Dataset loaded successfully. Shape: {frame.shape}")

    target_column = find_target_column(frame)
    target = normalize_binary_target(frame[target_column])
    raw_features = frame.drop(columns=[target_column])
    features = preprocess_features(raw_features)

    processed_target_name = "target"
    processed = features.copy()
    processed[processed_target_name] = target.to_numpy(dtype=np.int64)

    # Stratified round-robin assignment keeps both classes represented in all
    # five clients while producing non-overlapping client partitions.
    rng = np.random.default_rng(RANDOM_SEED)
    client_indices = {name: [] for name in CLIENT_NAMES}

    for class_value in sorted(processed[processed_target_name].unique()):
        class_indices = processed.index[
            processed[processed_target_name] == class_value
        ].to_numpy()
        rng.shuffle(class_indices)

        for position, row_index in enumerate(class_indices):
            client_name = CLIENT_NAMES[position % len(CLIENT_NAMES)]
            client_indices[client_name].append(int(row_index))

    print("[*] Writing five standardized client partitions:")
    for client_name in CLIENT_NAMES:
        indices = np.asarray(client_indices[client_name], dtype=np.int64)
        rng.shuffle(indices)
        partition = processed.loc[indices].reset_index(drop=True)

        output = source.parent / f"client_{client_name}_data.csv"
        partition.to_csv(output, index=False)

        counts = partition[processed_target_name].value_counts().to_dict()
        print(
            f"    -> {output.name}: rows={len(partition)}, "
            f"features={features.shape[1]}, classes={counts}"
        )

    metadata = {
        "source_file": source.name,
        "original_target_column": target_column,
        "target_column": processed_target_name,
        "feature_columns": list(features.columns),
        "feature_count": int(features.shape[1]),
        "client_names": list(CLIENT_NAMES),
        "preprocessing": {
            "numeric_missing_values": "global median",
            "categorical_missing_values": "global mode",
            "categorical_encoding": "one-hot",
            "feature_scaling": "global StandardScaler",
            "split_method": "class-stratified round robin",
            "random_seed": RANDOM_SEED,
        },
    }

    metadata_path = source.parent / "federated_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"[+] Metadata written to {metadata_path.name}")
    print("[SUCCESS] Real-data federated partitions are ready.")


if __name__ == "__main__":
    target_csv = sys.argv[1] if len(sys.argv) > 1 else "smoking.csv"
    split_kaggle_dataset(target_csv)
