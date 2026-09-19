# Feature extractors for sarcasm detection

This experiment extracts frozen Base and Large backbone features from English
MSTD++ and Chinese MCSD1 `train`, `valid`, and `test` CSVs. Run commands from
this directory. Labels are `0 = non-sarcastic` and `1 = sarcastic`. The scripts
read original transcripts, never zero-shot prompt JSONL.

## Layout and work plan

| Stage | Status | Work |
| --- | --- | --- |
| Data extract | Done | Validate split CSVs and media; extract BERT, Wav2Vec 2.0, and ResNet50 features from English MSTD++. |
| Data extract | Done | Extract LLaMA 3-8B, Qwen2-Audio-7B, and Qwen2.5-VL-7B features on English MSTD++; record exact pooling and image policy. |
| Data extract | Done | MCSD1 adapter for `File Name`, `Transcriptions`, `Labels` and its audio/video paths. |
| Model | Done | Collaborative-gate classifier adapted from the reference DNN for one, two, or three modalities. |
| Trainer | Done | KEY alignment, split checks, train-only normalization, warm-up, cosine decay, early stopping, checkpoints, and per-seed test reports. |

## Install

Use a PyTorch / torchvision pair built for the same CUDA or CPU environment.
Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

For English, the Base defaults are `bert-base-uncased`, `facebook/wav2vec2-base`, and
`torchvision`'s `ResNet50_Weights.IMAGENET1K_V2`. Model downloads require a
network connection on the first run. Use `--bert-model`, `--wav2vec2-model`,
and `--resnet-weights` for local model files. The ResNet weights file must be a
torchvision-compatible state dictionary. Chinese uses `bert-base-chinese` and
`TencentGameMate/chinese-wav2vec2-base`; ResNet50 is unchanged. Any model path
can be overridden on the command line. Large model defaults are the same across
both languages.

## Validate data and extract

```bash
python -m data_extract.base --extractor all --validate-only
python -m data_extract.base --extractor all --device cuda
```

Large backbones (each loads its own model and runs over all three splits):

```bash
python -m data_extract.large --extractor all --validate-only
python -m data_extract.large --extractor all --device cuda
```

Use `--extractor llama3`, `--extractor qwen2_audio`, or `--extractor qwen2_5_vl`
to run one model. The script uses the existing model directories under
`../../IS26/modelscope_cache/models/` when present; otherwise it uses their
Hugging Face IDs. Model paths can be overridden with `--llama3-model`,
`--qwen2-audio-model`, and `--qwen2-5-vl-model`. On CUDA, `--dtype auto` uses
bfloat16 when supported and float16 otherwise. `--batch-size` affects LLaMA
only. A GPU with enough memory to load one 7B/8B model is required for a
practical run.

To run one backbone or split, for example:

```bash
python -m data_extract.base --extractor bert --splits train valid test --device cuda
```

`--data-root` and `--output-dir` override the default paths. By default output
is `features/{dataset}/{extractor}/{train,valid,test}.npz`, with a
matching JSON metadata file. Re-running an extractor replaces its archives.

Every NPZ contains `keys` (MSTD++ `KEY` or MCSD1 `File Name`), `labels` (`int64`), and `features`
(`float32`). BERT and Wav2Vec 2.0 each yield `[N,768]`; ResNet50 yields
`[N,8,2048]`. Text pooling excludes tokenizer padding and includes model
special tokens. Audio is converted to mono and resampled to the model sample
rate. Utterances longer than 20 seconds are divided into near-equal chunks;
the feature is the mean over all valid contextual time steps from every chunk.
`--audio-chunk-seconds` changes this limit. This covers MCSD1 utterances longer
than Qwen2-Audio's single-segment input window. Eight video frames are sampled
uniformly, including the first and last frame; frame features are kept separate
for the future fusion model. The extractors run in evaluation and inference mode.

The Large shapes are `[N,4096]` for LLaMA 3 text and Qwen2-Audio, and
`[N,8,3584]` for Qwen2.5-VL. LLaMA 3 uses the mean of non-padding token states
from the final hidden layer, with transcript text only. Qwen2-Audio uses the
contextual audio tower output after its trained 1280-to-4096 multimodal
projector, averaged over valid time steps; it does not inject a text prompt or
run the language decoder. Qwen2.5-VL uses eight uniformly sampled frames,
processes each as an image with at most 50176 pixels, averages its vision
tokens, and retains one 3584-d vector per frame. The vision tower runs without
a text prompt; a prompt would only affect subsequent language decoder states.
These choices keep A and V inputs isolated and make the paper's stated feature
dimensions explicit. If the paper intends prompt-conditioned features, that
requires a separate decoder-state experiment and corresponding method text.

Example loader:

```python
import numpy as np

with np.load("features/mstdpp/bert/train.npz", allow_pickle=False) as data:
    keys, labels, features = data["keys"], data["labels"], data["features"]
```

See [`model/README.md`](model/README.md) and [`trainer/README.md`](trainer/README.md)
for the next-stage contracts.

## Train classifiers

```bash
python -m trainer.train --backbone base --modalities all --seeds 42 52 62 --device cuda
python -m trainer.train --backbone large --modalities T A T+A --seeds 42 52 62 --device cuda
```

`--backbone both` runs the same requested ablations for Base and Large.
`--modalities all` expands to T, A, V, T+A, T+V, A+V, and T+A+V.
Large experiments involving V require the Qwen2.5-VL archives. Missing
archives fail before training starts. See `python -m trainer.train --help` for
optimization and output options.

## Chinese MCSD1 workflow

MCSD1 uses `../raw_data/MCSD1/{train,valid,test}.csv`, where `s` maps to 1
and `ns` maps to 0. The media paths are `audios/{File Name}.wav` and
`videos/{File Name}.mp4`. Outputs go to `features/mcsd1` and `outputs/mcsd1`.

```bash
python -m data_extract.base --dataset mcsd1 --extractor all --validate-only
python -m data_extract.large --dataset mcsd1 --extractor all --validate-only
python -m data_extract.base --dataset mcsd1 --extractor all --device cuda
python -m data_extract.large --dataset mcsd1 --extractor all --device cuda
python -m trainer.train --dataset mcsd1 --backbone both --modalities all \
  --seeds 42 52 62 --device cuda --amp
```

Run extraction before training. `--extractor` and `--modalities` allow smaller
experiments; training reports missing feature archives before starting.
