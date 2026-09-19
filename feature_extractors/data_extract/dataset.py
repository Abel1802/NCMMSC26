"""Load original MSTD++ or MCSD1 splits without prompt text or label leakage."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "raw_data" / "mstdpp"
DATA_ROOTS = {
    "mstdpp": DEFAULT_DATA_ROOT,
    "mcsd1": Path(__file__).resolve().parents[2] / "raw_data" / "MCSD1",
}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Sample:
    key: str
    text: str
    label: int
    audio: Path
    video: Path


def load_split(data_root: Path, split: str, modality: str, dataset: str = "mstdpp") -> list[Sample]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if modality not in {"T", "A", "V"}:
        raise ValueError(f"Unknown modality: {modality}")
    if dataset not in DATA_ROOTS:
        raise ValueError(f"Unknown dataset: {dataset}")

    csv_path = data_root / f"{split}.csv"
    columns = ("KEY", "SENTENCE", "Sarcasm") if dataset == "mstdpp" else (
        "File Name", "Transcriptions", "Labels"
    )
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
        rows = list(reader)

    samples: list[Sample] = []
    seen: set[str] = set()
    for line, row in enumerate(rows, 2):
        key = row[columns[0]].strip()
        if not key or key in seen:
            raise ValueError(f"{csv_path}:{line}: empty or duplicate KEY {key!r}")
        seen.add(key)
        text = row[columns[1]].strip()
        if not text:
            raise ValueError(f"{csv_path}:{line}: empty {columns[1]}")
        if dataset == "mstdpp":
            try:
                label_value = float(row[columns[2]])
            except ValueError as exc:
                raise ValueError(f"{csv_path}:{line}: invalid Sarcasm label") from exc
            if label_value not in (0.0, 1.0):
                raise ValueError(f"{csv_path}:{line}: Sarcasm must be 0 or 1")
            label = int(label_value)
            audio_dir, video_dir = "final_utterance_audios", "final_utterance_videos"
        else:
            label_text = row[columns[2]].strip().lower()
            if label_text not in {"s", "ns"}:
                raise ValueError(f"{csv_path}:{line}: Labels must be s or ns")
            label = int(label_text == "s")
            audio_dir, video_dir = "audios", "videos"

        audio = data_root / audio_dir / f"{key}.wav"
        video = data_root / video_dir / f"{key}.mp4"
        media = {"A": audio, "V": video}.get(modality)
        if media is not None and not media.is_file():
            raise FileNotFoundError(f"{csv_path}:{line}: missing {media}")
        samples.append(Sample(key, text, label, audio, video))
    return samples


def load_splits(
    data_root: Path, splits: tuple[str, ...], modality: str, dataset: str = "mstdpp"
) -> dict[str, list[Sample]]:
    result = {split: load_split(data_root, split, modality, dataset) for split in splits}
    owners: dict[str, str] = {}
    for split, samples in result.items():
        for sample in samples:
            previous = owners.setdefault(sample.key, split)
            if previous != split:
                raise ValueError(f"KEY {sample.key!r} appears in both {previous} and {split}")
    return result
