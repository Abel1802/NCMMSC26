"""Summarize three-seed test results for MSTD++ and MCSD1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


DATASETS = (("mstdpp", "English (MSTD++)"), ("mcsd1", "Chinese (MCSD1)"))
MODALITIES = ("T", "A", "V", "T+A", "T+V", "A+V", "T+A+V")
BACKBONES = ("base", "large")
EXPECTED_SEEDS = (42, 52, 62)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root / "outputs")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    return parser.parse_args()


def modality_dir(modality: str) -> str:
    return modality.replace("+", "_")


def load_run(path: Path, dataset: str, backbone: str, modality: str, seed: int) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("backbone"), result.get("modalities"), result.get("seed")) != (backbone, modality, seed):
        raise ValueError(f"Metadata mismatch in {path}")
    test = result["test"]
    per_class = test["per_class"]
    precisions = [float(per_class[str(label)]["precision"]) for label in (0, 1)]
    recalls = [float(per_class[str(label)]["recall"]) for label in (0, 1)]
    return {
        "acc": float(test["accuracy"]),
        "macro_p": float(np.mean(precisions)),
        "macro_r": float(np.mean(recalls)),
        "macro_f1": float(test["macro_f1"]),
    }


def summarize(input_dir: Path) -> dict[str, list[dict]]:
    tables = {}
    for dataset, _ in DATASETS:
        rows = []
        for modality in MODALITIES:
            for backbone in BACKBONES:
                runs = []
                for seed in EXPECTED_SEEDS:
                    path = input_dir / dataset / backbone / modality_dir(modality) / f"seed_{seed}" / "result.json"
                    runs.append(load_run(path, dataset, backbone, modality, seed))
                row = {"modality": modality, "scale": backbone.capitalize(), "seeds": list(EXPECTED_SEEDS)}
                for metric in ("acc", "macro_p", "macro_r", "macro_f1"):
                    row[metric] = float(np.mean([run[metric] for run in runs]))
                rows.append(row)
        tables[dataset] = rows
    return tables


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("Modality", "Scale", "ACC", "Macro-P", "Macro-R", "Macro-F1", "Seeds"))
        for row in rows:
            writer.writerow((row["modality"], row["scale"], *(f"{100 * row[m]:.2f}" for m in
                              ("acc", "macro_p", "macro_r", "macro_f1")), "42,52,62"))


def markdown_table(rows: list[dict]) -> list[str]:
    lines = [
        "| Modality | Scale | ACC | Macro-P | Macro-R | Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [f"{100 * row[m]:.2f}" for m in ("acc", "macro_p", "macro_r", "macro_f1")]
        lines.append(f"| {row['modality']} | {row['scale']} | " + " | ".join(values) + " |")
    return lines


def main() -> None:
    args = parse_args()
    tables = summarize(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = [
        "# Three-seed feature-classifier results",
        "",
        "All values are test-set percentages averaged over seeds 42, 52, and 62. "
        "Macro-P and Macro-R are computed across classes within each seed before averaging across seeds.",
    ]
    for dataset, title in DATASETS:
        text.extend(("", f"## {title}", "", *markdown_table(tables[dataset])))
        write_csv(args.output_dir / f"{dataset}_three_seed_mean.csv", tables[dataset])
    (args.output_dir / "three_seed_mean_results.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
