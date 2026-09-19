#!/usr/bin/env python3
"""Build Chinese MCSD1 zero-shot True/False modality test sets for ms-swift."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_DIR / "raw_data/MCSD1/test.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "ms-swift/jsonl_data_test/mcsd1"
MODALITIES = ("T", "A", "V", "T+A", "T+V", "A+V", "T+A+V")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modality",
        choices=MODALITIES,
        action="append",
        help="Generate only the selected modality; repeat to select several. Default: all seven.",
    )
    return parser.parse_args()


def label_to_text(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if label == "s":
        return "True"
    if label == "ns":
        return "False"
    raise ValueError(f"Labels must be 's' or 'ns', got {raw_label!r}")


def make_prompt(transcription: str, modality: str) -> str:
    if modality == "T":
        task = (
            f'The following utterance is: "{transcription}".\n'
            "Analyze only the textual content of the utterance to determine whether the speaker is being sarcastic."
        )
    elif modality == "A":
        task = (
            "<audio>Analyze only the speaker's tone and speech prosody in the audio "
            "to determine whether the speaker is being sarcastic."
        )
    elif modality == "V":
        task = (
            "<video>Analyze only the speaker's facial expressions and visible movements in the video "
            "to determine whether the speaker is being sarcastic."
        )
    elif modality == "T+A":
        task = (
            f'<audio>The following utterance is: "{transcription}".\n'
            "Analyze the textual content, tone, and speech prosody to determine whether the speaker is being sarcastic."
        )
    elif modality == "T+V":
        task = (
            f'<video>The following utterance is: "{transcription}".\n'
            "Analyze the textual content, facial expressions, and visible movements "
            "to determine whether the speaker is being sarcastic."
        )
    elif modality == "A+V":
        task = (
            "<video><audio>Analyze the speaker's tone, speech prosody, facial expressions, and visible movements "
            "to determine whether the speaker is being sarcastic."
        )
    else:
        task = (
            f'<video><audio>The following utterance is: "{transcription}".\n'
            "Analyze the textual content, tone, speech prosody, facial expressions, and visible movements "
            "to determine whether the speaker is being sarcastic."
        )

    return (
        f"{task}\n"
        "Reason internally, but do not output your reasoning. Output exactly True if the utterance is sarcastic "
        "or False if it is not sarcastic.\n"
        "Your entire response must be exactly one word: True or False."
    )


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_dir = args.output_dir.resolve()
    modalities = tuple(args.modality) if args.modality else MODALITIES

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"File Name", "Transcriptions", "Labels"}
        missing_columns = required.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing CSV columns: {sorted(missing_columns)}")
        rows = list(reader)

    media_root = csv_path.parent
    for modality in modalities:
        use_audio = "A" in modality
        use_video = "V" in modality
        records = []
        missing_media = []

        for line_number, row in enumerate(rows, start=2):
            sample_id = row["File Name"].strip()
            transcription = row["Transcriptions"].strip()
            answer = label_to_text(row["Labels"])
            audio_path = (media_root / "audios" / f"{sample_id}.wav").resolve()
            video_path = (media_root / "videos" / f"{sample_id}.mp4").resolve()

            if use_audio and not audio_path.is_file():
                missing_media.append((line_number, audio_path))
            if use_video and not video_path.is_file():
                missing_media.append((line_number, video_path))

            record = {
                "messages": [
                    {"role": "user", "content": make_prompt(transcription, modality)},
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

        suffix = modality.replace("+", "_")
        output_path = output_dir / f"test_zh_zero_shot_bool_{suffix}.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Wrote {len(records)} {modality} records to {output_path}")


if __name__ == "__main__":
    main()
