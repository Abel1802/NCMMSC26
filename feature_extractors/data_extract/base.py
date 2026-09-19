"""Extract frozen Base backbone features for English MSTD++ or Chinese MCSD1."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .dataset import DATA_ROOTS, SPLITS, Sample, load_splits


SPECS = {
    "bert": ("T", 768, "bert-base-uncased"),
    "wav2vec2": ("A", 768, "facebook/wav2vec2-base"),
    "resnet50": ("V", 2048, "torchvision/ResNet50_Weights.IMAGENET1K_V2"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATA_ROOTS), default="mstdpp")
    parser.add_argument("--data-root", type=Path, help="Default: dataset-specific raw_data directory")
    parser.add_argument("--output-dir", type=Path, help="Default: features/{dataset}")
    parser.add_argument("--extractor", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=16, help="BERT sentence batch size")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--audio-chunk-seconds", type=float, default=20.0,
                        help="Maximum duration per audio encoder call; all chunks contribute to the mean")
    parser.add_argument("--bert-model", help="Default: bert-base-uncased (English) or bert-base-chinese (Chinese)")
    parser.add_argument("--wav2vec2-model", help="Default: language-specific Wav2Vec 2.0 Base checkpoint")
    parser.add_argument("--resnet-weights", type=Path, help="Optional local torchvision ResNet50 state_dict")
    parser.add_argument("--validate-only", action="store_true", help="Check CSVs and required media without loading models")
    return parser.parse_args()


def extract_bert(samples: list[Sample], model_id: str, device: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            inputs = tokenizer(
                [sample.text for sample in batch], padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            vectors.append(pooled.float().cpu().numpy())
    return np.concatenate(vectors) if vectors else np.empty((0, 768), dtype=np.float32)


def load_audio(path: Path, target_rate: int) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    signal, rate = sf.read(path, dtype="float32", always_2d=True)
    if signal.shape[0] == 0:
        raise ValueError(f"Empty audio: {path}")
    mono = signal.mean(axis=1)
    if rate != target_rate:
        divisor = math.gcd(rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, rate // divisor)
    return np.asarray(mono, dtype=np.float32)


def split_audio(waveform: np.ndarray, rate: int, max_seconds: float) -> list[np.ndarray]:
    """Use near-equal chunks so the final segment is never extremely short."""
    count = max(1, math.ceil(len(waveform) / (rate * max_seconds)))
    return list(np.array_split(waveform, count))


def extract_wav2vec2(samples: list[Sample], model_id: str, device: str, chunk_seconds: float) -> np.ndarray:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id).to(device).eval()
    rate = processor.sampling_rate
    vectors = []
    with torch.inference_mode():
        for sample in samples:
            waveform = load_audio(sample.audio, rate)
            total, steps = None, 0
            for chunk in split_audio(waveform, rate, chunk_seconds):
                # One segment at a time: no padding frames enter the temporal mean.
                inputs = processor(chunk, sampling_rate=rate, return_tensors="pt")
                inputs = {name: value.to(device) for name, value in inputs.items()}
                hidden = model(**inputs).last_hidden_state
                segment_sum = hidden.float().sum(dim=1).squeeze(0).cpu()
                total = segment_sum if total is None else total + segment_sum
                steps += hidden.shape[1]
            vectors.append((total / steps).numpy())
    return np.stack(vectors) if vectors else np.empty((0, 768), dtype=np.float32)


def sample_video_frames(path: Path, count: int = 8):
    from decord import VideoReader, cpu
    from PIL import Image

    reader = VideoReader(str(path), ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError(f"Empty video: {path}")
    indices = np.rint(np.linspace(0, len(reader) - 1, count)).astype(np.int64)
    return [Image.fromarray(frame) for frame in reader.get_batch(indices.tolist()).asnumpy()]


def extract_resnet50(samples: list[Sample], device: str, weights_path: Path | None) -> np.ndarray:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=None if weights_path else weights)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.fc = torch.nn.Identity()  # global average pool output: 2048 per frame
    model = model.to(device).eval()
    preprocess = weights.transforms()
    vectors = []
    with torch.inference_mode():
        for sample in samples:
            frames = sample_video_frames(sample.video)
            batch = torch.stack([preprocess(frame) for frame in frames]).to(device)
            vectors.append(model(batch).float().cpu().numpy())
    return np.stack(vectors) if vectors else np.empty((0, 8, 2048), dtype=np.float32)


def save_features(path: Path, samples: list[Sample], vectors: np.ndarray, metadata: dict) -> None:
    if len(vectors) != len(samples) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid feature output for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(
            temporary, keys=np.asarray([s.key for s in samples]),
            labels=np.asarray([s.label for s in samples], dtype=np.int64),
            features=vectors.astype(np.float32, copy=False),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root or DATA_ROOTS[args.dataset]
    args.output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "features" / args.dataset
    args.bert_model = args.bert_model or ("bert-base-chinese" if args.dataset == "mcsd1" else SPECS["bert"][2])
    args.wav2vec2_model = args.wav2vec2_model or (
        "TencentGameMate/chinese-wav2vec2-base" if args.dataset == "mcsd1" else SPECS["wav2vec2"][2]
    )
    if args.batch_size < 1 or args.max_text_length < 2 or args.audio_chunk_seconds < 1:
        raise ValueError("Invalid batch size, text length, or audio chunk duration")
    names = tuple(SPECS) if args.extractor == "all" else (args.extractor,)
    splits = tuple(dict.fromkeys(args.splits))
    loaded = {}
    for name in names:
        modality, _, _ = SPECS[name]
        loaded[name] = load_splits(args.data_root, splits, modality, args.dataset)
        print(f"{name}: " + ", ".join(f"{split}={len(loaded[name][split])}" for split in splits), flush=True)
    if args.validate_only:
        return

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    models = {"bert": args.bert_model, "wav2vec2": args.wav2vec2_model,
              "resnet50": str(args.resnet_weights) if args.resnet_weights else SPECS["resnet50"][2]}
    for name in names:
        modality, dim, _ = SPECS[name]
        for split in splits:
            samples = loaded[name][split]
            print(f"Extracting {name}/{split} on {device}...", flush=True)
            if name == "bert":
                features = extract_bert(samples, args.bert_model, device, args.batch_size, args.max_text_length)
            elif name == "wav2vec2":
                features = extract_wav2vec2(samples, args.wav2vec2_model, device, args.audio_chunk_seconds)
            else:
                features = extract_resnet50(samples, device, args.resnet_weights)
            expected = (len(samples), 8, dim) if modality == "V" else (len(samples), dim)
            if features.shape != expected:
                raise ValueError(f"{name}/{split}: got {features.shape}, expected {expected}")
            path = args.output_dir / name / f"{split}.npz"
            metadata = {
                "dataset": "MCSD1" if args.dataset == "mcsd1" else "MSTD++",
                "split": split, "extractor": name,
                "modality": modality, "model": models[name], "shape": list(features.shape),
                "pooling": "masked_token_mean" if name == "bert" else
                           "time_mean" if name == "wav2vec2" else "8_uniform_frames_global_average_pool",
                "label_mapping": {"0": "non-sarcastic", "1": "sarcastic"},
                "source_csv": str((args.data_root / f"{split}.csv").resolve()),
            }
            if name == "bert":
                metadata["max_text_length"] = args.max_text_length
            if name == "wav2vec2":
                metadata["audio_chunk_seconds"] = args.audio_chunk_seconds
            save_features(path, samples, features, metadata)
            print(f"Saved {path}: {features.shape}", flush=True)


if __name__ == "__main__":
    main()
