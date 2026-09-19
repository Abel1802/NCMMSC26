# NCMMSC26 Spoken Sarcasm Detection

本目录包含英文 MSTD++ 与中文 MCSD1 的口语讽刺识别实验，覆盖：

- MLLM zero-shot 与 SFT；
- Base/Large 冻结特征提取；
- T、A、V、T+A、T+V、A+V、T+A+V 七种模态消融；
- 协同门控分类器训练及三随机种子结果汇总。

## 数据与标签

`raw_data` 是指向共享数据目录的符号链接。

| 数据集 | 语言 | Train | Valid | Test | 原始标签 |
|---|---|---:|---:|---:|---|
| MSTD++ | 英文 | 841 | 180 | 181 | `1.0/0.0` |
| MCSD1 | 中文 | 1893 | 406 | 406 | `s/ns` |

统一使用 `1/True = sarcasm`、`0/False = non-sarcasm`。媒体路径为：

- MSTD++：`raw_data/mstdpp/final_utterance_{audios,videos}/`；
- MCSD1：`raw_data/MCSD1/{audios,videos}/`。

## 目录结构

```text
NCMMSC26/
├── raw_data -> shared dataset directory
├── src/                         # zero-shot JSONL 生成
├── ms-swift/                    # zero-shot、SFT、推理及评估
└── feature_extractors/
    ├── data_extract/            # Base/Large 特征提取
    ├── model/                   # collaborative-gate classifier
    ├── trainer/                 # 训练与结果统计
    ├── features/{mstdpp,mcsd1}/
    └── outputs/{mstdpp,mcsd1}/
```

## 模态约束

| 模态 | Transcript | WAV | Video |
|---|---:|---:|---:|
| T | ✓ |  |  |
| A |  | ✓ |  |
| V |  |  | ✓ |
| T+A | ✓ | ✓ |  |
| T+V | ✓ |  | ✓ |
| A+V |  | ✓ | ✓ |
| T+A+V | ✓ | ✓ | ✓ |

视频可能包含音轨。所有 V 相关实验必须设置 `USE_AUDIO_IN_VIDEO=false`；A+V 和
T+A+V 仅使用独立 WAV 作为音频，避免模态泄漏与重复音频。

## Zero-shot

从本目录生成七种模态的测试 JSONL：

```bash
python src/prepare_mstdpp_zero_shot.py --all-ablation-modalities
python src/prepare_mstdpp_zero_shot.py --modality 'T+A+V'
python src/prepare_mcsd1_zero_shot.py
```

输出位于 `ms-swift/jsonl_data_test/{mstdpp,mcsd1}/`。Prompt 只提供指定模态，要求模型
严格输出 `True` 或 `False`。运行 Qwen2.5-Omni 示例：

```bash
cd ms-swift
CUDA_VISIBLE_DEVICES=0 \
DATASET_PATH="$PWD/jsonl_data_test/mcsd1/test_zh_zero_shot_bool_T_A_V.jsonl" \
RESULT_PATH="$PWD/zero_shot_results/mcsd1/qwen2_5_omni_7b_T_A_V.jsonl" \
bash run_mstdpp_zero_shot_qwen2_5_omni.sh
```

运行脚本使用确定性解码，并关闭 audio output 和 video 内音轨。结果可通过
`evaluate_zero_shot_results.py` 评估。其他推理与 SFT 入口见 `ms-swift/run_*.sh` 和
`ms-swift/*.sbatch`。

## Base/Large 特征

| 模态 | Base | Large | 输出形状 |
|---|---|---|---|
| T | BERT | LLaMA 3-8B | `[N,768]` / `[N,4096]` |
| A | Wav2Vec 2.0 | Qwen2-Audio-7B | `[N,768]` / `[N,4096]` |
| V | ResNet50 | Qwen2.5-VL-7B | `[N,8,2048]` / `[N,8,3584]` |

每个 NPZ 保存 `keys`、`labels` 和 `features`。音频超过 20 秒时分块处理并对所有有效
时间步汇总；视频均匀采样 8 帧。详细定义见 `feature_extractors/README.md`。

```bash
cd feature_extractors
python -m pip install -r requirements.txt

# 英文 MSTD++
python -m data_extract.base  --dataset mstdpp --extractor all --device cuda
python -m data_extract.large --dataset mstdpp --extractor all --device cuda

# 中文 MCSD1
python -m data_extract.base  --dataset mcsd1 --extractor all --device cuda
python -m data_extract.large --dataset mcsd1 --extractor all --device cuda
```

中文 Base 文本与音频默认使用 `bert-base-chinese` 和
`TencentGameMate/chinese-wav2vec2-base`；模型路径均可通过命令行覆盖。

## 分类器训练

模型参考 `mlt_sarcasm/src/SVM_DNN/run_dnn.py` 的
`Speaker_Independent_Triple_Mode_without_Context`，支持单、双、三模态。训练包含
train-only 标准化、AdamW、warm-up、cosine decay、梯度裁剪、early stopping、验证集
选模和多随机种子汇总，测试集不参与选模。

```bash
cd feature_extractors
python -m trainer.train --dataset mstdpp --backbone both --modalities all \
  --seeds 42 52 62 --device cuda --amp
python -m trainer.train --dataset mcsd1 --backbone both --modalities all \
  --seeds 42 52 62 --device cuda --amp
```

已有目录默认不会覆盖；重新运行需显式传入 `--overwrite`。

## 结果汇总

```bash
cd feature_extractors
python -m trainer.summarize_results
```

测试集三种子均值见：

- `feature_extractors/outputs/three_seed_mean_results.md`；
- `feature_extractors/outputs/mstdpp_three_seed_mean.csv`；
- `feature_extractors/outputs/mcsd1_three_seed_mean.csv`。

汇总指标为 ACC、Macro-P、Macro-R 和 Macro-F1；随机种子为 42、52、62。
