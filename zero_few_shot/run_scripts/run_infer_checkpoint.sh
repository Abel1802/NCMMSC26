CUDA_VISIBLE_DEVICES=0 \
ENABLE_AUDIO_OUTPUT=false \
USE_AUDIO_IN_VIDEO=false \
FPS=2 FPS_MAX_FRAMES=12 VIDEO_MAX_TOKEN_NUM=256 \
swift infer \
  --adapters sft_outputs/mcsd1_qwen2_5_omni_T_A_V_1000/v0-20260918-112917/checkpoint-189 \
  --val_dataset jsonl_data_test/mcsd1/test_zh_zero_shot_bool_T_A_V.jsonl \
  --result_path sft_outputs/mcsd1_qwen2_5_omni_T_A_V_1000/v0-20260918-112917/checkpoint-189_test.jsonl \
  --infer_backend transformers \
  --torch_dtype bfloat16 \
  --max_batch_size 1 \
  --max_new_tokens 1 \
  --temperature 0 \
  --stream false \
  --metric acc