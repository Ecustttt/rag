#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/deploy_chat.sh"
"$SCRIPT_DIR/deploy_emb.sh"

echo "两个服务已启动"
