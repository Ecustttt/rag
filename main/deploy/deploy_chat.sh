#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ---- 默认值 ----
CHAT_MODEL="${CHAT_MODEL:-$HOME/models/Qwen3.5-4B-AWQ-4bit}"
#CHAT_MODEL="${CHAT_MODEL:-$HOME/models/Qwen3.5-9B-AWQ}"
CHAT_PORT="${CHAT_PORT:-8000}"
#CHAT_SERVED_NAME="${CHAT_SERVED_NAME:-qwen3.5-4b}"
CHAT_SERVED_NAME="${CHAT_SERVED_NAME:-chat_model}"
CHAT_MAX_LEN="${CHAT_MAX_LEN:-4096}"
CHAT_MAX_SEQS="${CHAT_MAX_SEQS:-2}"
CHAT_GPU_UTIL="${CHAT_GPU_UTIL:-0.70}"
CHAT_LOG="$LOG_DIR/vllm-chat.log"

echo "启动 chat 服务: $CHAT_SERVED_NAME @ $CHAT_PORT"
vllm serve "$CHAT_MODEL" \
  --host 0.0.0.0 --port "$CHAT_PORT" \
  --trust-remote-code --dtype auto \
  --max-model-len "$CHAT_MAX_LEN" \
  --max-num-seqs "$CHAT_MAX_SEQS" \
  --gpu-memory-utilization "$CHAT_GPU_UTIL" \
  --enforce-eager \
  --language-model-only \
  --served-model-name "$CHAT_SERVED_NAME" \
  > "$CHAT_LOG" 2>&1 &

echo "等待 chat 服务就绪 ..."
wait_ready "http://127.0.0.1:$CHAT_PORT/health" 60 3 "$CHAT_LOG"