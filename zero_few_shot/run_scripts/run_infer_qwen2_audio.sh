#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${CHECKPOINT:-${SCRIPT_DIR}/sft_outputs/mstdpp_qwen2_audio_A/v1-20260917-163119/checkpoint-633}"
TEST_DATASET="${TEST_DATASET:-${SCRIPT_DIR}/jsonl_data_test/mstdpp/test_en_zero_shot_bool_A.jsonl}"
RESULT_PATH="${RESULT_PATH:-${SCRIPT_DIR}/sft_outputs/mstdpp_qwen2_audio_A/checkpoint-633_test_max1.jsonl}"

if [[ ! -f "${CHECKPOINT}/adapter_config.json" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 1
fi
if [[ ! -f "${TEST_DATASET}" ]]; then
    echo "Test dataset not found: ${TEST_DATASET}" >&2
    exit 1
fi
if [[ -e "${RESULT_PATH}" ]]; then
    echo "Result already exists (swift infer appends to it): ${RESULT_PATH}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
swift infer \
    --adapters "${CHECKPOINT}" \
    --val_dataset "${TEST_DATASET}" \
    --result_path "${RESULT_PATH}" \
    --infer_backend transformers \
    --torch_dtype bfloat16 \
    --max_batch_size 1 \
    --max_new_tokens 1 \
    --temperature 0 \
    --stream false \
    --metric acc
