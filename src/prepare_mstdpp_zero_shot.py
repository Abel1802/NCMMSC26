#!/usr/bin/env python3
"""Build MSTD++ zero-shot True/False modality test sets for ms-swift."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_DIR / "raw_data/mstdpp/test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "ms-swift/jsonl_data_test/mstdpp"
MODALITIES = ("T", "A", "V", "T+A", "T+V", "A+V", "T+A+V")
ABLATION_MODALITIES = MODALITIES[:-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modality", choices=MODALITIES, default="T+A+V")
    parser.add_argument(
        "--all-ablation-modalities",
        action="store_true",
        help="Generate T, A, V, T+A, T+V, and A+V in one invocation.",
    )
    return parser.parse_args()


def label_to_text(raw_label: str) -> str:
    try:
        label = float(raw_label)
    except ValueError as exc:
        raise ValueError(f"Invalid Sarcasm label: {raw_label!r}") from exc
    if label == 1.0:
        return "True"
    if label == 0.0:
        return "False"
    raise ValueError(f"Sarcasm label must be 0 or 1, got {raw_label!r}")


def make_prompt(sentence: str, modality: str) -> str:
    if modality == "T":
        prefix = f'The following utterance is: "{sentence}".\n'
        evidence = "Analyze only the textual content of the utterance"
    elif modality == "A":
        prefix = "<audio>"
        evidence = "Analyze only the speaker's tone and speech prosody in the audio"
    elif modality == "V":
        prefix = "<video>"
        evidence = "Analyze only the speaker's facial expressions and visible movements in the video"
    elif modality == "T+A":
        prefix = f'<audio>The following utterance is: "{sentence}".\n'
        evidence = "Analyze the textual content, tone, and speech prosody"
    elif modality == "T+V":
        prefix = f'<video>The following utterance is: "{sentence}".\n'
        evidence = "Analyze the textual content, facial expressions, and visible movements"
    elif modality == "A+V":
        prefix = "<video><audio>"
        evidence = "Analyze the speaker's tone, speech prosody, facial expressions, and visible movements"
    else:
        prefix = f'<video><audio>The following utterance is: "{sentence}".\n'
        evidence = "Analyze the speaker's words, tone, prosody, and facial expressions"

    return (
        f"{prefix}{evidence} to determine whether the speaker is being sarcastic.\n"
        "Reason internally, but do not output your reasoning. Output exactly "
        "True if the utterance is sarcastic or False if it is not sarcastic.\n"
        "Your entire response must be exactly one word: True or False."
    )


def output_name(modality: str) -> str:
    if modality == "T+A+V":
        return "test_en_zero_shot_bool.jsonl"
    return f"test_en_zero_shot_bool_{modality.replace('+', '_')}.jsonl"


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()
    media_root = csv_path.parent

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"KEY", "SENTENCE", "Sarcasm"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
        rows = list(reader)

    modalities = ABLATION_MODALITIES if args.all_ablation_modalities else (args.modality,)
    for modality in modalities:
        records = []
        missing_media = []
        use_audio = "A" in modality
        use_video = "V" in modality

        for line_number, row in enumerate(rows, start=2):
            key = row["KEY"].strip()
            sentence = row["SENTENCE"].strip()
            answer = label_to_text(row["Sarcasm"])
            audio_path = (media_root / "final_utterance_audios" / f"{key}.wav").resolve()
            video_path = (media_root / "final_utterance_videos" / f"{key}.mp4").resolve()

            required_media = []
            if use_audio:
                required_media.append(audio_path)
            if use_video:
                required_media.append(video_path)
            for media_path in required_media:
                if not media_path.is_file():
                    missing_media.append((line_number, media_path))

            record = {
                "messages": [
                    {"role": "user", "content": make_prompt(sentence, modality)},
                    {"role": "assistant", "content": answer},
                ],
                "label": answer,
            }
            if use_audio:
                record["audios"] = [str(audio_path)]
            if use_video:
                record["videos"] = [str(video_path)]
            records.append(record)

        if missing_media:
            details = "\n".join(f"CSV line {line}: {path}" for line, path in missing_media[:20])
            raise FileNotFoundError(
                f"Found {len(missing_media)} missing media files for {modality}. "
                f"First entries:\n{details}"
            )

        output_path = output_dir / output_name(modality)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {len(records)} {modality} records to {output_path}")


if __name__ == "__main__":
    main()
