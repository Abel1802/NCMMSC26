"""Validate and align Base/Large feature archives by dataset sample key."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


EXTRACTORS = {
    "base": {"T": "bert", "A": "wav2vec2", "V": "resnet50"},
    "large": {"T": "llama3", "A": "qwen2_audio", "V": "qwen2_5_vl"},
}
SPLITS = ("train", "valid", "test")


@dataclass
class SplitData:
    keys: list[str]
    labels: np.ndarray
    features: dict[str, np.ndarray]


def _read_archive(path: Path, modality: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"keys", "labels", "features"}:
            raise ValueError(f"{path}: expected keys, labels, features")
        keys = archive["keys"].tolist()
        labels = archive["labels"]
        features = archive["features"]
    if not keys or len(keys) != len(set(keys)) or any(not key for key in keys):
        raise ValueError(f"{path}: empty or duplicate keys")
    if labels.shape != (len(keys),) or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path}: invalid labels")
    if features.shape[0] != len(keys) or not np.isfinite(features).all():
        raise ValueError(f"{path}: invalid features")
    if modality == "V":
        if features.ndim != 3 or features.shape[1] != 8:
            raise ValueError(f"{path}: expected video shape [N,8,D]")
        features = features.mean(axis=1)
    elif features.ndim != 2:
        raise ValueError(f"{path}: expected shape [N,D]")
    if features.shape[1] < 1:
        raise ValueError(f"{path}: feature dimension must be positive")
    return keys, labels.astype(np.int64), features.astype(np.float32)


def load_data(feature_root: Path, backbone: str, modalities: tuple[str, ...]) -> dict[str, SplitData]:
    if backbone not in EXTRACTORS or not modalities or set(modalities) - set("TAV"):
        raise ValueError("Invalid backbone or modalities")
    result = {}
    dimensions = {}
    split_owners: dict[str, str] = {}
    for split in SPLITS:
        canonical_keys = None
        canonical_labels = None
        arrays = {}
        for modality in modalities:
            extractor = EXTRACTORS[backbone][modality]
            path = feature_root / extractor / f"{split}.npz"
            keys, labels, features = _read_archive(path, modality)
            if canonical_keys is None:
                canonical_keys, canonical_labels = keys, labels
            else:
                if set(keys) != set(canonical_keys):
                    raise ValueError(f"{path}: KEY set differs from other modalities in {split}")
                positions = {key: index for index, key in enumerate(keys)}
                order = [positions[key] for key in canonical_keys]
                labels, features = labels[order], features[order]
                if not np.array_equal(labels, canonical_labels):
                    raise ValueError(f"{path}: labels disagree across modalities")
            if modality in dimensions and features.shape[1] != dimensions[modality]:
                raise ValueError(f"{path}: feature dimension differs across splits")
            dimensions[modality] = features.shape[1]
            arrays[modality] = features
        assert canonical_keys is not None and canonical_labels is not None
        for key in canonical_keys:
            previous = split_owners.setdefault(key, split)
            if previous != split:
                raise ValueError(f"KEY {key!r} appears in {previous} and {split}")
        result[split] = SplitData(canonical_keys, canonical_labels, arrays)
    return result


def normalize_from_train(splits: dict[str, SplitData]) -> dict[str, dict[str, np.ndarray]]:
    """Standardize pooled features; fit mean and scale on train only."""
    statistics = {}
    for modality, training in splits["train"].features.items():
        mean = training.mean(axis=0, dtype=np.float64)
        scale = training.std(axis=0, dtype=np.float64)
        scale = np.where(scale < 1e-6, 1.0, scale)
        statistics[modality] = {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32)}
        for split in SPLITS:
            value = (splits[split].features[modality] - mean) / scale
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite normalized {modality} features in {split}")
            splits[split].features[modality] = value.astype(np.float32)
    return statistics


class FeatureDataset(Dataset):
    def __init__(self, split: SplitData) -> None:
        self.keys = split.keys
        self.labels = torch.from_numpy(split.labels)
        self.features = {m: torch.from_numpy(value) for m, value in split.features.items()}

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        return self.keys[index], {m: value[index] for m, value in self.features.items()}, self.labels[index]
