#!/usr/bin/env bash
#SBATCH --job-name=mcsd1-size-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=12:00:00
#SBATCH --chdir=/gpfs/work3/0/prjs0864/phd_projects/NCMMSC26/ms-swift
#SBATCH --output=/gpfs/work3/0/prjs0864/phd_projects/NCMMSC26/ms-swift/sft_outputs/mcsd1-size-sft-%j.log

set -euo pipefail

SCRIPT_DIR="/gpfs/work3/0/prjs0864/phd_projects/NCMMSC26/ms-swift"
DATA_DIR="${SCRIPT_DIR}/jsonl_data_sft/mcsd1"
MODEL_DIR="/gpfs/work3/0/prjs0864/phd_projects/IS26/modelscope_cache/models"
DRY_RUN="${DRY_RUN:-0}"

[[ -d "${SCRIPT_DIR}" ]] || { echo "Project directory not found: ${SCRIPT_DIR}" >&2; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "Model directory not found: ${MODEL_DIR}" >&2; exit 1; }

if [[ "${DRY_RUN}" != 1 ]]; then
    module load 2025
    module load CUDA/12.8.0
    source activate ms2
    export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    command -v swift >/dev/null || { echo 'swift is unavailable in the ms2 environment' >&2; exit 1; }
fi

cd "${SCRIPT_DIR}"
export NPROC_PER_NODE=1
export ENABLE_AUDIO_OUTPUT=false
export USE_AUDIO_IN_VIDEO=false
export FPS=2
export FPS_MAX_FRAMES=12
export VIDEO_MAX_TOKEN_NUM=256

run_case() {
    local size="$1" modality="$2"
    local model model_type template model_name
    local gradient_accumulation_steps=4 eval_steps=100 save_steps=100
    case "${modality}" in
        T)
            model="${MODEL_DIR}/LLM-Research/Meta-Llama-3-8B-Instruct"
            model_type=llama
            template=llama3
            model_name=llama3
            ;;
        A)
            model="${MODEL_DIR}/Qwen/Qwen2-Audio-7B-Instruct"
            model_type=qwen2_audio
            template=qwen2_audio
            model_name=qwen2_audio
            ;;
        V)
            model="${MODEL_DIR}/Qwen/Qwen2___5-VL-7B-Instruct"
            model_type=qwen2_5_vl
            template=qwen2_5_vl
            model_name=qwen2_5_vl
            ;;
        T_A_V)
            model="${MODEL_DIR}/Qwen/Qwen2___5-Omni-7B"
            model_type=qwen2_5_omni
            template=qwen2_5_omni
            model_name=qwen2_5_omni
            # The existing full-data Omni run used 4 GPUs x accumulation 4.
            # On one GPU, accumulation 16 preserves its effective batch size.
            gradient_accumulation_steps=16
            eval_steps=50
            save_steps=50
            ;;
        *) echo "Unknown modality: ${modality}" >&2; exit 1 ;;
    esac

    local train_data="${DATA_DIR}/train_zh_sft_bool_${modality}_${size}.jsonl"
    local valid_data="${DATA_DIR}/valid_zh_sft_bool_${modality}.jsonl"
    local output_dir="${SCRIPT_DIR}/sft_outputs/mcsd1_${model_name}_${modality}_${size}"

    for path in "${model}" "${train_data}" "${valid_data}"; do
        [[ -e "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 1; }
    done
    if [[ -f "${output_dir}/.sft_complete" && "${DRY_RUN}" != 1 ]]; then
        echo "Skipping completed run: ${output_dir}"
        return
    fi

    local -a cmd=(
        swift sft
        --model "${model}"
        --model_type "${model_type}"
        --template "${template}"
        --dataset "${train_data}"
        --val_dataset "${valid_data}"
        --tuner_type lora
        --torch_dtype bfloat16
        --num_train_epochs 3
        --per_device_train_batch_size 1
        --per_device_eval_batch_size 1
        --gradient_accumulation_steps "${gradient_accumulation_steps}"
        --learning_rate 1e-4
        --seed 42
        --data_seed 42
        --lora_rank 8
        --lora_alpha 32
        --target_modules all-linear
        --freeze_vit true
        --freeze_aligner true
        --eval_steps "${eval_steps}"
        --save_steps "${save_steps}"
        --save_total_limit 2
        --logging_steps 5
        --max_length 4096
        --warmup_ratio 0.05
        --dataloader_num_workers 4
        --deepspeed zero2
        --output_dir "${output_dir}"
    )

    echo "[$(date '+%F %T')] Starting ${size} samples, ${modality}: ${output_dir}"
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
        return
    fi
    "${cmd[@]}"
    touch "${output_dir}/.sft_complete"
    echo "[$(date '+%F %T')] Completed ${size} samples, ${modality}"
}

for size in 500 1000; do
    for modality in T A V T_A_V; do
        run_case "${size}" "${modality}"
    done
done
