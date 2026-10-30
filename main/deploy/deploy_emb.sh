#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"


# ---- 默认值 ----
EMB_MODEL="${EMB_MODEL:-$HOME/models/Qwen3-Embedding-0.6B}"
EMB_PORT="${EMB_PORT:-8001}"
#EMB_SERVED_NAME="${EMB_SERVED_NAME:-qwen3-embedding-0.6b}"
EMB_SERVED_NAME="${EMB_SERVED_NAME:-embedding_model}"
EMB_MAX_LEN="${EMB_MAX_LEN:-800}"
EMB_MAX_SEQS="${EMB_MAX_SEQS:-4}"
EMB_GPU_UTIL="${EMB_GPU_UTIL:-0.20}"
EMB_LOG="$LOG_DIR/vllm-emb.log"

echo "启动 embedding 服务: $EMB_SERVED_NAME @ $EMB_PORT"
vllm serve "$EMB_MODEL" \
  --host 0.0.0.0 --port "$EMB_PORT" \
  --trust-remote-code --dtype auto \
  --max-model-len "$EMB_MAX_LEN" \
  --max-num-seqs "$EMB_MAX_SEQS" \
  --gpu-memory-utilization "$EMB_GPU_UTIL" \
  --served-model-name "$EMB_SERVED_NAME" \
  > "$EMB_LOG" 2>&1 &

echo "等待 emb 服务就绪 ..."
wait_ready "http://127.0.0.1:$EMB_PORT/health" 60 3 "$EMB_LOG"