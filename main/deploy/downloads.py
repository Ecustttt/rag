#!/usr/bin/env python3
"""从 ModelScope 下载模型"""

import os
import sys
from pathlib import Path

# 检查依赖
try:
    from modelscope import snapshot_download
except ImportError:
    print("未检测到 modelscope 库，正在安装...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope", "-q"])
    from modelscope import snapshot_download

# ---- 配置 ----
#MODEL_REPO = "tclf90/Qwen3.5-9B-AWQ"
#MODEL_DIR = str(Path.home() / "models" / "Qwen3.5-9B-AWQ")
MODEL_REPO = "OpenDataLab/MinerU2.5-Pro-2605-1.2B"
MODEL_DIR = str(Path.home() / "models" / "MinerU2.5-Pro-2605-1.2B")


def main():
    print(f"准备下载模型: {MODEL_REPO}")
    print(f"目标目录: {MODEL_DIR}")
    print()

    # 创建目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("开始下载，请稍候...")
    print("(首次下载可能需要几分钟，请耐心等待)")
    print()

    try:
        local_dir = snapshot_download(
            model_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            #allow_patterns=["*.json"],
        )
        print()
        print("✅ 模型下载完成！")
        print(f"模型路径: {local_dir}")
        print()

        # 列出文件
        files = list(Path(local_dir).iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"共 {len(files)} 个文件，总大小约 {total_size / 1024**3:.2f} GB")
        print()
        print("你可以用以下命令启动服务：")
        print(f"  bash start_chat.sh  # 假设你已有启动脚本")

    except Exception as e:
        print(f"❌ 下载失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()