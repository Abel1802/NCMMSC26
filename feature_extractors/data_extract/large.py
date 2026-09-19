"""Extract frozen Large backbone features for English MSTD++ or Chinese MCSD1."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

from .base import load_audio, sample_video_frames, save_features, split_audio
from .dataset import DATA_ROOTS, SPLITS, Sample, load_splits


MODEL_ROOT = Path(__file__).resolve().parents[3] / "IS26/modelscope_cache/models"
SPECS = {
    "llama3": ("T", 4096, MODEL_ROOT / "LLM-Research/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct"),
    "qwen2_audio": ("A", 4096, MODEL_ROOT / "Qwen/Qwen2-Audio-7B", "Qwen/Qwen2-Audio-7B"),
    "qwen2_5_vl": ("V", 3584, MODEL_ROOT / "Qwen/Qwen2___5-VL-7B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"),
}


def default_model(name: str) -> str:
    _, _, local, hub = SPECS[name]
    return str(local if (local / "config.json").is_file() else hub)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATA_ROOTS), default="mstdpp")
    parser.add_argument("--data-root", type=Path, help="Default: dataset-specific raw_data directory")
    parser.add_argument("--output-dir", type=Path, help="Default: features/{dataset}")
    parser.add_argument("--extractor", choices=(*SPECS, "all"), default="all")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--llama3-model", default=default_model("llama3"))
    parser.add_argument("--qwen2-audio-model", default=default_model("qwen2_audio"))
    parser.add_argument("--qwen2-5-vl-model", default=default_model("qwen2_5_vl"))
    parser.add_argument("--batch-size", type=int, default=4, help="LLaMA text batch size")
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--audio-chunk-seconds", type=float, default=20.0,
                        help="Maximum duration per audio encoder call; all chunks contribute to the mean")
    parser.add_argument("--max-image-pixels", type=int, default=50176,
                        help="Qwen2.5-VL maximum pixels per sampled frame")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def choose_device_and_dtype(device_arg: str, dtype_arg: str):
    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else device_arg
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if dtype_arg == "auto":
        dtype_arg = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
            "float16" if device.startswith("cuda") else "float32"
        )
    if device == "cpu" and dtype_arg == "float16":
        raise ValueError("float16 on CPU is unsupported for this pipeline")
    return device, getattr(torch, dtype_arg)


def load_llama(model_id: str, device: str, dtype):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    if model.config.hidden_size != 4096:
        raise ValueError(f"Expected LLaMA hidden size 4096, got {model.config.hidden_size}")
    return tokenizer, model


def extract_llama(samples: list[Sample], loaded, device: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    tokenizer, model = loaded
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            inputs = tokenizer(
                [s.text for s in samples[start:start + batch_size]],
                padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs, use_cache=False).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
    return np.concatenate(vectors) if vectors else np.empty((0, 4096), np.float32)


def load_qwen_audio(model_id: str, device: str, dtype):
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    if model.multi_modal_projector.linear.out_features != 4096:
        raise ValueError("Qwen2-Audio projector does not output 4096-dimensional features")
    return processor, model


def extract_qwen_audio(samples: list[Sample], loaded, device: str, chunk_seconds: float) -> np.ndarray:
    import torch

    processor, model = loaded
    rate = processor.feature_extractor.sampling_rate
    vectors = []
    with torch.inference_mode():
        for sample in samples:
            waveform = load_audio(sample.audio, rate)
            total, steps = None, 0
            for chunk in split_audio(waveform, rate, chunk_seconds):
                inputs = processor.feature_extractor(
                    chunk, sampling_rate=rate, return_attention_mask=True,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                mel = inputs["input_features"].to(device=device, dtype=model.audio_tower.conv1.weight.dtype)
                mask = inputs["attention_mask"].to(device)
                feature_lengths, output_lengths = model.audio_tower._get_feat_extract_output_lengths(mask.sum(-1))
                max_length = (mel.shape[-1] - 2) // 2 + 1
                padded = torch.arange(max_length, device=device)[None, :] >= feature_lengths[:, None]
                attention = padded[:, None, None, :].expand(1, 1, max_length, max_length)
                attention = attention.to(dtype=mel.dtype)
                attention = attention.masked_fill(padded[:, None, None, :], float("-inf"))
                contextual = model.audio_tower(mel, attention_mask=attention).last_hidden_state
                projected = model.multi_modal_projector(contextual)
                valid = int(output_lengths[0].item())
                if valid < 1 or valid > projected.shape[1]:
                    raise ValueError(f"Invalid Qwen2-Audio output length for {sample.key}: {valid}")
                segment_sum = projected[0, :valid].float().sum(0).cpu()
                total = segment_sum if total is None else total + segment_sum
                steps += valid
            vectors.append((total / steps).numpy())
    return np.stack(vectors) if vectors else np.empty((0, 4096), np.float32)


def load_qwen_vl(model_id: str, device: str, dtype):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    if model.config.vision_config.out_hidden_size != 3584:
        raise ValueError("Qwen2.5-VL vision tower does not output 3584-dimensional features")
    return processor, model


def extract_qwen_vl(samples: list[Sample], loaded, device: str, max_pixels: int) -> np.ndarray:
    import torch

    processor, model = loaded
    vectors = []
    with torch.inference_mode():
        for sample in samples:
            frame_vectors = []
            for frame in sample_video_frames(sample.video, count=8):
                inputs = processor.image_processor(
                    images=frame, max_pixels=max_pixels, return_tensors="pt",
                )
                pixels = inputs["pixel_values"].to(
                    device=device, dtype=model.model.visual.patch_embed.proj.weight.dtype
                )
                grid = inputs["image_grid_thw"].to(device)
                tokens = model.model.visual(pixels, grid_thw=grid)
                if tokens.shape[1] != 3584:
                    raise ValueError(f"Unexpected Qwen2.5-VL token shape: {tuple(tokens.shape)}")
                frame_vectors.append(tokens.float().mean(0).cpu().numpy())
            vectors.append(np.stack(frame_vectors))
    return np.stack(vectors) if vectors else np.empty((0, 8, 3584), np.float32)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root or DATA_ROOTS[args.dataset]
    args.output_dir = args.output_dir or Path(__file__).resolve().parents[1] / "features" / args.dataset
    if args.batch_size < 1 or args.max_text_length < 2 or args.max_image_pixels < 3136 or args.audio_chunk_seconds < 1:
        raise ValueError("Invalid batch size, text length, image pixel limit, or audio chunk duration")
    names = tuple(SPECS) if args.extractor == "all" else (args.extractor,)
    if "qwen2_audio" in names and args.audio_chunk_seconds > 30:
        raise ValueError("Qwen2-Audio accepts at most about 30 seconds per segment; use <=30")
    splits = tuple(dict.fromkeys(args.splits))
    loaded_data = {}
    for name in names:
        modality = SPECS[name][0]
        loaded_data[name] = load_splits(args.data_root, splits, modality, args.dataset)
        print(f"{name}: " + ", ".join(f"{s}={len(loaded_data[name][s])}" for s in splits), flush=True)
    if args.validate_only:
        return

    import torch

    device, dtype = choose_device_and_dtype(args.device, args.dtype)
    models = {"llama3": args.llama3_model, "qwen2_audio": args.qwen2_audio_model,
              "qwen2_5_vl": args.qwen2_5_vl_model}
    loaders = {"llama3": load_llama, "qwen2_audio": load_qwen_audio, "qwen2_5_vl": load_qwen_vl}
    for name in names:
        print(f"Loading {name}: {models[name]}", flush=True)
        loaded_model = loaders[name](models[name], device, dtype)
        modality, dimension, _, _ = SPECS[name]
        for split in splits:
            samples = loaded_data[name][split]
            print(f"Extracting {name}/{split} on {device} ({dtype})...", flush=True)
            if name == "llama3":
                features = extract_llama(samples, loaded_model, device, args.batch_size, args.max_text_length)
            elif name == "qwen2_audio":
                features = extract_qwen_audio(samples, loaded_model, device, args.audio_chunk_seconds)
            else:
                features = extract_qwen_vl(samples, loaded_model, device, args.max_image_pixels)
            shape = (len(samples), 8, dimension) if modality == "V" else (len(samples), dimension)
            if features.shape != shape:
                raise ValueError(f"{name}/{split}: got {features.shape}, expected {shape}")
            output = args.output_dir / name / f"{split}.npz"
            metadata = {
                "dataset": "MCSD1" if args.dataset == "mcsd1" else "MSTD++",
                "split": split, "extractor": name, "modality": modality,
                "model": models[name], "shape": list(features.shape), "dtype": "float32",
                "inference_dtype": str(dtype),
                "pooling": {"llama3": "last_layer_masked_token_mean",
                            "qwen2_audio": "audio_tower_projector_valid_time_mean",
                            "qwen2_5_vl": "8_uniform_frames_vision_token_mean"}[name],
                "label_mapping": {"0": "non-sarcastic", "1": "sarcastic"},
                "source_csv": str((args.data_root / f"{split}.csv").resolve()),
            }
            if name == "llama3":
                metadata["max_text_length"] = args.max_text_length
            if name == "qwen2_5_vl":
                metadata["max_image_pixels"] = args.max_image_pixels
            if name == "qwen2_audio":
                metadata["audio_chunk_seconds"] = args.audio_chunk_seconds
            save_features(output, samples, features, metadata)
            print(f"Saved {output}: {features.shape}", flush=True)
        del loaded_model
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
