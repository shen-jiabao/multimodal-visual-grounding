#!/usr/bin/env bash
# ============================================================
# 下载模型权重到 checkpoints/（走 HF 国内镜像，存在即跳过）
#
# 用法：
#   bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct
#   bash scripts/download_model.sh IDEA-Research/grounding-dino-base checkpoints/GroundingDINObase
#
# 关键坑（来自《云端部署记录》）：
#   - hf-mirror 的 CAS 不支持 xet，必须 HF_HUB_DISABLE_XET=1，否则大文件 401 失败
#   - 新版 huggingface_hub 用 `hf` 命令（旧 huggingface-cli 已废弃）
# ============================================================
set -euo pipefail

# 非交互 shell 下 conda 的 python/pip/hf 可能不在 PATH，补上
for _p in /root/miniconda3/bin /opt/conda/bin; do
  [ -d "$_p" ] && PATH="$_p:$PATH"
done
export PATH

REPO="${1:?用法: bash scripts/download_model.sh <repo_id> [dest_dir]}"
DEST="${2:-checkpoints/$(basename "$REPO")}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="$HF_ENDPOINT"
export HF_HUB_DISABLE_XET=1
PYPI="${PYPI:-https://pypi.tuna.tsinghua.edu.cn/simple}"

echo "== HF_ENDPOINT=$HF_ENDPOINT  HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
echo "== 目标：repo=$REPO  dest=$DEST"

if ls "$DEST"/*.safetensors >/dev/null 2>&1 && [ -f "$DEST/config.json" ]; then
  echo "== 已存在，跳过下载：$DEST"
  exit 0
fi

mkdir -p "$DEST"
echo "== 安装/更新 huggingface_hub[cli] ..."
pip install --no-input -U "huggingface_hub[cli]" -i "$PYPI" >/dev/null

echo "== 开始下载（排除 *.bin）..."
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$DEST" --exclude "*.bin"
else
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$REPO", local_dir="$DEST", ignore_patterns=["*.bin"])
PY
fi

echo "== 完成。目录内容："
ls -lh "$DEST" | sed -n '1,40p'
