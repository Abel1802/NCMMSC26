"""Train collaborative-gate classifiers on MSTD++ or MCSD1 Base/Large features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from model.collaborative_gate import CollaborativeGateClassifier
from trainer.data import EXTRACTORS, FeatureDataset, load_data, normalize_from_train


ROOT = Path(__file__).resolve().parents[1]
MODALITY_OPTIONS = ("T", "A", "V", "T+A", "T+V", "A+V", "T+A+V")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mstdpp", "mcsd1"), default="mstdpp")
    parser.add_argument("--feature-root", type=Path, help="Default: features/{dataset}")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/{dataset}")
    parser.add_argument("--backbone", choices=("base", "large", "both"), default="base")
    parser.add_argument("--modalities", nargs="+", choices=(*MODALITY_OPTIONS, "all"), default=["T+A+V"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--shared-dim", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="none")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing seed directories")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (args.batch_size < 1 or args.epochs < 1 or args.min_epochs < 1 or args.patience < 1
            or args.num_workers < 0 or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError("Invalid batch size, epoch/patience value, workers, or duplicate seed")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.max_grad_norm < 0:
        raise ValueError("Invalid learning rate, weight decay, or gradient clipping limit")
    if not 0 <= args.warmup_ratio < 1 or not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("Warmup and minimum LR ratios must be in [0,1]")
    if not 0 <= args.label_smoothing < 1 or args.min_delta < 0:
        raise ValueError("Invalid label smoothing or minimum improvement")
    if args.shared_dim < 1 or args.projection_dim < 1 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid model dimensions or dropout")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(split, batch_size: int, shuffle: bool, seed: int, workers: int, pin_memory: bool):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        FeatureDataset(split), batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=pin_memory, generator=generator,
    )


def to_device(features, device):
    return {modality: values.to(device, non_blocking=True) for modality, values in features.items()}


def metrics(labels: list[int], predictions: list[int]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1.mean()),
        "binary_f1": float(f1[1]),
        "per_class": {
            str(index): {"precision": float(precision[index]), "recall": float(recall[index]),
                         "f1": float(f1[index]), "support": int(support[index])}
            for index in (0, 1)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    keys, labels, predictions, probabilities = [], [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch_keys, features, targets in loader:
            features = to_device(features, device)
            targets = targets.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss")
            prob = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            total_loss += loss.item() * len(targets)
            keys.extend(batch_keys)
            labels.extend(targets.cpu().tolist())
            predictions.extend(prob.argmax(axis=1).tolist())
            probabilities.extend(prob.tolist())
    result = metrics(labels, predictions)
    result["loss"] = total_loss / len(labels)
    return result, list(zip(keys, labels, predictions, probabilities))


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_predictions(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("KEY", "label", "prediction", "prob_non_sarcastic", "prob_sarcastic"))
        for key, label, prediction, probability in rows:
            writer.writerow((key, label, prediction, probability[0], probability[1]))


def save_checkpoint(path: Path, state: dict) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train_one(args, backbone: str, modality_name: str, splits, stats, seed: int, device: torch.device):
    set_seed(seed)
    modalities = tuple(modality_name.split("+"))
    output = args.output_dir / backbone / modality_name.replace("+", "_") / f"seed_{seed}"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(splits["train"], args.batch_size, True, seed, args.num_workers, device.type == "cuda")
    valid_loader = make_loader(splits["valid"], args.batch_size, False, seed, args.num_workers, device.type == "cuda")
    test_loader = make_loader(splits["test"], args.batch_size, False, seed, args.num_workers, device.type == "cuda")

    dimensions = {modality: splits["train"].features[modality].shape[1] for modality in modalities}
    model = CollaborativeGateClassifier(
        dimensions, shared_dim=args.shared_dim, projection_dim=args.projection_dim, dropout=args.dropout,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    class_weights = None
    if args.class_weight == "balanced":
        counts = np.bincount(splits["train"].labels, minlength=2)
        if np.any(counts == 0):
            raise ValueError("Both classes must be present for balanced class weights")
        class_weights = torch.tensor(len(splits["train"].labels) / (2 * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(int(round(args.warmup_ratio * total_steps)), total_steps - 1)

    def lr_multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * (1 + math.cos(math.pi * min(1, progress))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration.update({
        "backbone_run": backbone, "modalities_run": modality_name, "seed_run": seed,
        "extractors": {m: EXTRACTORS[backbone][m] for m in modalities},
        "dimensions": dimensions, "parameter_count": parameter_count,
        "train_count": len(splits["train"].keys), "valid_count": len(splits["valid"].keys),
        "test_count": len(splits["test"].keys), "warmup_steps": warmup_steps,
    })
    configuration["feature_metadata"] = {}
    for modality in modalities:
        metadata_path = args.feature_root / EXTRACTORS[backbone][modality] / "train.json"
        if metadata_path.is_file():
            configuration["feature_metadata"][modality] = json.loads(metadata_path.read_text(encoding="utf-8"))
    save_json(output / "config.json", configuration)
    np.savez_compressed(output / "train_normalization.npz", **{
        f"{m}_{field}": value for m, values in stats.items() for field, value in values.items()
    })
    best_f1, best_loss, best_epoch = -1.0, float("inf"), 0
    bad_epochs, global_step = 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for _, features, targets in train_loader:
            features = to_device(features, device)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(features)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            if args.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
                global_step += 1
            total_train_loss += loss.item() * len(targets)

        validation, _ = evaluate(model, valid_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": total_train_loss / len(splits["train"].keys),
               "valid_loss": validation["loss"], "valid_macro_f1": validation["macro_f1"],
               "valid_accuracy": validation["accuracy"], "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"{backbone} {modality_name} seed={seed} epoch={epoch}: "
              f"train_loss={row['train_loss']:.4f} valid_loss={row['valid_loss']:.4f} "
              f"valid_macro_f1={row['valid_macro_f1']:.4f}", flush=True)
        improved = best_epoch == 0 or validation["macro_f1"] > best_f1 + args.min_delta
        tied_but_lower_loss = (abs(validation["macro_f1"] - best_f1) <= args.min_delta
                               and validation["loss"] < best_loss)
        if improved or tied_but_lower_loss:
            best_f1, best_loss, best_epoch = validation["macro_f1"], validation["loss"], epoch
            save_checkpoint(output / "best.pt", {
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "valid_macro_f1": best_f1, "valid_loss": best_loss,
                "dimensions": dimensions, "modalities": modalities,
            })
        bad_epochs = 0 if improved else bad_epochs + 1
        if epoch >= args.min_epochs and global_step >= warmup_steps and bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    save_json(output / "history.json", history)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation, valid_rows = evaluate(model, valid_loader, criterion, device)
    test, test_rows = evaluate(model, test_loader, criterion, device)
    save_predictions(output / "valid_predictions.csv", valid_rows)
    save_predictions(output / "test_predictions.csv", test_rows)
    result = {"backbone": backbone, "modalities": modality_name, "seed": seed,
              "best_epoch": best_epoch, "epochs_run": len(history), "parameter_count": parameter_count,
              "validation": validation, "test": test}
    save_json(output / "result.json", result)
    print(f"RESULT {backbone} {modality_name} seed={seed}: "
          f"val_f1={validation['macro_f1']:.4f} test_f1={test['macro_f1']:.4f} "
          f"test_acc={test['accuracy']:.4f}", flush=True)
    return result


def main() -> None:
    args = parse_args()
    args.feature_root = args.feature_root or ROOT / "features" / args.dataset
    args.output_dir = args.output_dir or ROOT / "outputs" / args.dataset
    validate_args(args)
    if "all" in args.modalities:
        if len(args.modalities) != 1:
            raise ValueError("Use --modalities all by itself")
        modality_names = MODALITY_OPTIONS
    else:
        modality_names = tuple(dict.fromkeys(args.modalities))
    backbones = ("base", "large") if args.backbone == "both" else (args.backbone,)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")

    # Validate every requested archive before starting any training run.
    for backbone in backbones:
        for name in modality_names:
            for modality in name.split("+"):
                for split in ("train", "valid", "test"):
                    path = args.feature_root / EXTRACTORS[backbone][modality] / f"{split}.npz"
                    if not path.is_file():
                        raise FileNotFoundError(f"Missing {backbone} {name} feature archive: {path}")
            for seed in args.seeds:
                output = args.output_dir / backbone / name.replace("+", "_") / f"seed_{seed}"
                if output.exists() and not args.overwrite:
                    raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    for backbone in backbones:
        for name in modality_names:
            modalities = tuple(name.split("+"))
            splits = load_data(args.feature_root, backbone, modalities)
            stats = normalize_from_train(splits)
            results = [train_one(args, backbone, name, splits, stats, seed, device) for seed in args.seeds]
            directory = args.output_dir / backbone / name.replace("+", "_")
            summary = {"backbone": backbone, "modalities": name, "seeds": list(args.seeds), "runs": results}
            for metric in ("accuracy", "macro_f1", "binary_f1"):
                values = np.array([run["test"][metric] for run in results], dtype=float)
                summary[f"test_{metric}_mean"] = float(values.mean())
                summary[f"test_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            save_json(directory / "summary.json", summary)


if __name__ == "__main__":
    main()
