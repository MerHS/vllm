MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="lmarena-ai/VisionArena-Chat"
NUM_PROMPTS=1000
LOG_PATH="results"


python ../benchmark_serving.py \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --model $MODEL \
  --dataset-name hf \
  --dataset-path $DATASET \
  --hf-split train \
  --num-prompts $NUM_PROMPTS \
  --seed 40 \
  --save-result \
  --save-detailed \
  --result-dir $LOG_PATH/vision_arena_results \
  --result-filename vision_arena_outputs$(date +"%Y%m%d_%H%M%S").json \
  --port 8000 | tee $LOG_PATH/benchmark_VisionArena_$(date +"%Y%m%d_%H%M%S").log 2>&1