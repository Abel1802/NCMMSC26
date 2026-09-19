# NCMMSC26 Spoken Sarcasm Detection

This directory contains spoken sarcasm detection experiments on English MSTD++
and Chinese MCSD1. It covers:

- MLLM zero-shot evaluation and supervised fine-tuning (SFT);
- frozen Base and Large feature extraction;
- seven modality settings: T, A, V, T+A, T+V, A+V, and T+A+V;
- collaborative-gate classifier training and three-seed result aggregation.

## Data and labels

`raw_data` is a symbolic link to the shared dataset directory.

| Dataset | Language | Train | Valid | Test | Original labels |
|---|---|---:|---:|---:|---|
| MSTD++ | English | 841 | 180 | 181 | `1.0/0.0` |
| MCSD1 | Chinese | 1893 | 406 | 406 | `s/ns` |

All experiments use `1/True = sarcasm` and `0/False = non-sarcasm`. Media is stored in:

- MSTD++: `raw_data/mstdpp/final_utterance_{audios,videos}/`;
- MCSD1: `raw_data/MCSD1/{audios,videos}/`.

## Directory layout

```text
NCMMSC26/
├── raw_data -> shared dataset directory
├── zero_few_shot/                         # zero-shot JSONL generation
├── ms-swift/                    # zero-shot, SFT, inference, and evaluation
└── feature_extractors/
    ├── data_extract/            # Base/Large feature extraction
    ├── model/                   # collaborative-gate classifier
    ├── trainer/                 # training and result aggregation
    ├── features/{mstdpp,mcsd1}/
    └── outputs/{mstdpp,mcsd1}/
```

## Modality constraints

| Modality | Transcript | WAV | Video |
|---|---:|---:|---:|
| T | ✓ |  |  |
| A |  | ✓ |  |
| V |  |  | ✓ |
| T+A | ✓ | ✓ |  |
| T+V | ✓ |  | ✓ |
| A+V |  | ✓ | ✓ |
| T+A+V | ✓ | ✓ | ✓ |

Video files may contain audio tracks. Every V experiment must set
`USE_AUDIO_IN_VIDEO=false`. A+V and T+A+V use only the separate WAV file as
audio input, preventing duplicate audio and modality leakage.

## Zero-shot evaluation

Generate test JSONL files for all seven modality settings from this directory:

```bash
python zero_few_shot/prepare_mstdpp_zero_shot.py --all-ablation-modalities
python zero_few_shot/prepare_mstdpp_zero_shot.py --modality 'T+A+V'
python zero_few_shot/prepare_mcsd1_zero_shot.py
```

Files are written to `ms-swift/jsonl_data_test/{mstdpp,mcsd1}/`. Each prompt
contains only the selected modalities and requires exactly `True` or `False`.
Example Qwen2.5-Omni invocation:

```bash
cd ms-swift
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH="$PWD/jsonl_data_test/mcsd1/test_zh_zero_shot_bool_T_A_V.jsonl" \
RESULT_PATH="$PWD/zero_shot_results/mcsd1/qwen2_5_omni_7b_T_A_V.jsonl" \
bash run_mstdpp_zero_shot_qwen2_5_omni.sh
```

The script uses deterministic decoding and disables audio output and the video
audio track. Evaluate results with `evaluate_zero_shot_results.py`. Other
inference and SFT entry points are in `ms-swift/run_*.sh` and `ms-swift/*.sbatch`.

## Base and Large features

| Modality | Base | Large | Output shape |
|---|---|---|---|
| T | BERT | LLaMA 3-8B | `[N,768]` / `[N,4096]` |
| A | Wav2Vec 2.0 | Qwen2-Audio-7B | `[N,768]` / `[N,4096]` |
| V | ResNet50 | Qwen2.5-VL-7B | `[N,8,2048]` / `[N,8,3584]` |

Each NPZ archive stores `keys`, `labels`, and `features`. Audio longer than 20
seconds is divided into chunks and pooled over all valid contextual time steps.
Video extraction uniformly samples eight frames. See `feature_extractors/README.md`
for the complete feature definitions.

```bash
cd feature_extractors
python -m pip install -r requirements.txt

# English MSTD++
python -m data_extract.base  --dataset mstdpp --extractor all --device cuda
python -m data_extract.large --dataset mstdpp --extractor all --device cuda

# Chinese MCSD1
python -m data_extract.base  --dataset mcsd1 --extractor all --device cuda
python -m data_extract.large --dataset mcsd1 --extractor all --device cuda
```

Chinese Base text and audio default to `bert-base-chinese` and
`TencentGameMate/chinese-wav2vec2-base`. Every model path can be overridden on
the command line.

## Classifier training

Training includes train-only normalization, AdamW, warm-up, cosine decay,
gradient clipping, early stopping, validation-based selection, and multi-seed
aggregation. The test set is never used for model selection.

```bash
cd feature_extractors
python -m trainer.train --dataset mstdpp --backbone both --modalities all \
  --seeds 42 52 62 --device cuda --amp
python -m trainer.train --dataset mcsd1 --backbone both --modalities all \
  --seeds 42 52 62 --device cuda --amp
```

Existing run directories are protected from accidental replacement. Pass
`--overwrite` explicitly to rerun them.

## Result aggregation

```bash
cd feature_extractors
python -m trainer.summarize_results
```

Three-seed test means are available in:

- `feature_extractors/outputs/three_seed_mean_results.md`;
- `feature_extractors/outputs/mstdpp_three_seed_mean.csv`;
- `feature_extractors/outputs/mcsd1_three_seed_mean.csv`.

Reported metrics are ACC, Macro-P, Macro-R, and Macro-F1 over seeds 42, 52, and 62.
