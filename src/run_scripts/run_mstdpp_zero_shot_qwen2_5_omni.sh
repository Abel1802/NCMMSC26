#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 {qwen2_5_vl|qwen2_5_omni|qwen2_audio|llama3} TEST_JSONL" >&2
    exit 2
fi

MODEL_NAME="$1"
TEST_JSONL="$(realpath -m -- "$2")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CACHE="${SCRIPT_DIR}/../../IS26/modelscope_cache/models/Qwen"
MAX_NEW_TOKENS=4

case "${MODEL_NAME}" in
    qwen2_5_vl)
        MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
        MODEL_TYPE=qwen2_5_vl
        TEMPLATE=qwen2_5_vl
        ;;
    qwen2_5_omni)
        MODEL="Qwen/Qwen2.5-Omni-7B"
        MODEL_TYPE=qwen2_5_omni
        TEMPLATE=qwen2_5_omni
        ;;
    qwen2_audio)
        MODEL="Qwen/Qwen2-Audio-7B-Instruct"
        MODEL_TYPE=qwen2_audio
        TEMPLATE=qwen2_audio
        MAX_NEW_TOKENS=1
        ;;
    llama3)
        MODEL="LLM-Research/Meta-Llama-3-8B-Instruct"
        MODEL_TYPE=llama
        TEMPLATE=llama3
        ;;
    *)
        echo "Unknown model: ${MODEL_NAME}" >&2
        exit 2
        ;;
esac

if [[ ! -f "${TEST_JSONL}" ]]; then
    echo "Test file not found: ${TEST_JSONL}" >&2
    exit 1
fi
if [[ "${MODEL}" == /* && ! -d "${MODEL}" ]]; then
    echo "Model directory not found: ${MODEL}" >&2
    exit 1
fi

RESULT_PATH="${RESULT_PATH:-${SCRIPT_DIR}/zero_shot_results/${MODEL_NAME}_$(basename -- "${TEST_JSONL}")}"
if [[ -e "${RESULT_PATH}" ]]; then
    echo "Result file already exists: ${RESULT_PATH}" >&2
    exit 1
fi
mkdir -p "$(dirname -- "${RESULT_PATH}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export ENABLE_AUDIO_OUTPUT=false
export USE_AUDIO_IN_VIDEO=false
export FPS=2
export FPS_MAX_FRAMES=12
export VIDEO_MAX_TOKEN_NUM=256

EXTRA_ARGS=()
if [[ "${MODEL_NAME}" == llama3 ]]; then
    EXTRA_ARGS+=(--enable_thinking false)
fi

swift infer \
    --model "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --template "${TEMPLATE}" \
    --infer_backend transformers \
    --device_map auto \
    --torch_dtype bfloat16 \
    --val_dataset "${TEST_JSONL}" \
    --result_path "${RESULT_PATH}" \
    --max_batch_size 1 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature 0 \
    --stream false \
    --val_dataset_shuffle false \
    --metric acc \
    "${EXTRA_ARGS[@]}"
