#!/bin/bash
# 公共环境 & 函数，被其他脚本 source

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

export VLLM_USE_MODELSCOPE=true
export VLLM_USE_FLASHINFER_SAMPLER=0

export no_proxy="127.0.0.1,localhost,0.0.0.0,::1"
export NO_PROXY="$no_proxy"

export CUDA_HOME=~/miniconda3/envs/rag/lib/python3.11/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"

# 日志目录
LOG_DIR="${LOG_DIR:-$HOME/logs}"
mkdir -p "$LOG_DIR"

# 轮询等待 HTTP 服务就绪
# 用法: wait_ready <url> <max_tries> <interval_sec> <log_file>
wait_ready() {
    local url="$1"
    local max_tries="${2:-60}"
    local interval="${3:-3}"
    local log_file="${4:-}"

    for i in $(seq 1 "$max_tries"); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "✅ $url 已就绪"
            return 0
        fi
        sleep "$interval"
    done

    echo "❌ 启动超时: $url"
    [ -n "$log_file" ] && echo "请检查日志: $log_file"
    return 1
}