#!/usr/bin/env bash
set -euo pipefail

SWIFT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${SWIFT_PROJECT_DIR}/jsonl_data_sft/mstdpp/train_en_sft_bool_T_A_V_500.jsonl}"
VAL_DATASET_PATH="${VAL_DATASET_PATH:-${SWIFT_PROJECT_DIR}/jsonl_data_sft/mstdpp/valid_en_sft_bool_T_A_V.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SWIFT_PROJECT_DIR}/sft_outputs/mstdpp_qwen2_5_omni_T_A_V_500}"

cd "${SWIFT_PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export ENABLE_AUDIO_OUTPUT=false
export USE_AUDIO_IN_VIDEO=false
export FPS=2
export FPS_MAX_FRAMES=12
export VIDEO_MAX_TOKEN_NUM=256

swift sft \
    --model "Qwen/Qwen2.5-Omni-7B" \
    --model_type qwen2_5_omni \
    --template qwen2_5_omni \
    --dataset "${TRAIN_DATASET_PATH}" \
    --val_dataset "${VAL_DATASET_PATH}" \
    --tuner_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --eval_steps 100 \
    --max_new_tokens 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --output_dir "${OUTPUT_DIR}"
