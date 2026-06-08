import json
from pathlib import Path
from typing import Dict

import numpy as np


ACCESS_FEATURE_DIM = 5
PAGE_FEATURE_DIM = 4

SEQUENCE_LENGTH = 32
FUTURE_WINDOW = 100


def load_trace(path: str):
    path_obj = Path(path)

    if path_obj.suffix == ".npz":
        data = np.load(path_obj)
        return {key: data[key] for key in data.files}

    with path_obj.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def future_reuse_within_window(vpns: np.ndarray, window: int = FUTURE_WINDOW) -> np.ndarray:
    labels = np.zeros(len(vpns), dtype=np.float32)

    for i in range(len(vpns)):
        end = min(len(vpns), i + window + 1)
        labels[i] = 1.0 if np.any(vpns[i + 1:end] == vpns[i]) else 0.0

    return labels


def preprocess(trace: Dict[str, np.ndarray]):

    vpn = trace["vpn"].astype(np.float32)

    delta = (
        trace["delta"]
        if "delta" in trace
        else trace["access_delta"]
    ).astype(np.float32)

    access = (
        trace["access"]
        if "access" in trace
        else trace["access_type"]
    )

    if access.dtype.kind in {"U", "S", "O"}:
        access = np.array(
            [1.0 if value == "write" else 0.0 for value in access],
            dtype=np.float32
        )
    else:
        access = access.astype(np.float32)

    timestamp = (
        trace["ts"]
        if "ts" in trace
        else trace["timestamp"]
    ).astype(np.float32)

    phase = (
        trace["phase"]
        if "phase" in trace
        else np.zeros(len(vpn), dtype=np.float32)
    ).astype(np.float32)

    access_features = np.stack(
        [vpn, delta, access, timestamp, phase],
        axis=1
    )

    labels = future_reuse_within_window(vpn)

    first_seen = {}
    last_access = {}
    frequency = {}

    page_features = np.zeros((len(vpn), PAGE_FEATURE_DIM), dtype=np.float32)

    for idx in range(len(vpn)):
        page = int(vpn[idx])

        if page not in first_seen:
            first_seen[page] = idx

        age = idx - last_access[page] if page in last_access else idx
        freq = frequency.get(page, 0)
        dirty = access[idx]
        residency_time = idx - first_seen[page]

        page_features[idx] = [float(age), float(freq), float(dirty), float(residency_time)]

        last_access[page] = idx
        frequency[page] = freq + 1

    return {
        "access_features": access_features,
        "page_features": page_features,
        "labels": labels
    }


def make_training_samples(
    access_features,
    page_features,
    labels,
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = 1
):

    sequences = []
    candidates = []
    targets = []

    for end in range(sequence_length, len(access_features), stride):
        start = end - sequence_length

        sequences.append(access_features[start:end])
        candidates.append(page_features[end])
        targets.append(labels[end])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(candidates, dtype=np.float32),
        np.asarray(targets, dtype=np.float32)
    )


def train_validation_split(
    sequences,
    page_features,
    labels,
    validation_ratio: float = 0.2
):

    split = int(len(labels) * (1.0 - validation_ratio))

    return (
        sequences[:split],
        page_features[:split],
        labels[:split],
        sequences[split:],
        page_features[split:],
        labels[split:]
    )


def save_numpy(
    output_path: str,
    sequences,
    page_features,
    labels
):

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        sequences=sequences,
        page_features=page_features,
        labels=labels
    )


def load_numpy(path: str):

    data = np.load(path)

    return (
        data["sequences"],
        data["page_features"],
        data["labels"]
    )
