#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."   # 回到项目根

DATASETS=("GWM" "MDM")
#MODES=("vector" "bm25" "hybrid")
#DATASETS=("MDM")
MODES=("bm25")

for ds in "${DATASETS[@]}"; do
    # 按 dataset 选后端端口
    if [ "$ds" = "GWM" ]; then
        PORT=8002
    else
        PORT=8003
    fi
    API="http://localhost:${PORT}/ask"

    for mode in "${MODES[@]}"; do
        echo ""
        echo "=============================="
        echo "  数据集: $ds  模式: $mode"
        echo "=============================="
        python eval/eval_ragas.py \
            --dataset "$ds" \
            --rag-api "$API" \
            --mode "$mode"
    done
done

echo ""
echo "全部完成"