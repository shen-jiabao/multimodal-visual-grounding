#!/usr/bin/env bash
# ============================================================
# AutoDL / Linux GPU 一键环境准备脚本
#
# 用法（在 AutoDL 实例终端中，先 cd 到项目根目录）：
#     bash scripts/setup_autodl.sh
#
# 完成三件事：
#   1. 检测已装 torch（AutoDL 镜像通常预装，避免重装破坏 CUDA 匹配）
#   2. 安装其余 Python 依赖（清华源加速，不重装 torch/torchvision）
#   3. 配置 HuggingFace 国内镜像，并下载 Grounding DINO base 模型到 checkpoints/
# ============================================================
set -euo pipefail

PYPI="https://pypi.tuna.tsinghua.edu.cn/simple"

echo "==> [1/4] 检查 Python 版本"
python --version
python -c "import sys; assert sys.version_info >= (3, 8), '需要 Python 3.8+'"

echo "==> [2/4] 检查 torch / CUDA 环境"
python - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
print("  cuda version:", torch.version.cuda if torch.cuda.is_available() else "N/A")
if torch.cuda.is_available():
    print("  GPU:", torch.cuda.get_device_name(0))
PY

echo "==> [3/4] 安装其余依赖（不重装 torch/torchvision）"
pip install --no-input \
  "transformers>=4.40" "accelerate>=0.26" "tokenizers>=0.19" "safetensors>=0.4" \
  "opencv-python-headless>=4.8" "Pillow>=9.0" "numpy>=1.24" "pandas>=1.5" \
  "matplotlib>=3.5" "seaborn>=0.12" "PyYAML>=6.0" "tqdm>=4.66" \
  -i "$PYPI"

echo "==> [4/4] 配置 HF 镜像并下载模型"
# 国内加速：使 huggingface_hub / transformers 走镜像
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com

MODEL_DIR="checkpoints/GroundingDINObase"
if [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/pytorch_model.bin" ]; then
  echo "  模型已存在，跳过下载：$MODEL_DIR"
else
  echo "  开始下载 Grounding DINO base（hf-mirror 镜像，仅 safetensors 权重）..."
  pip install --no-input -U "huggingface_hub[cli]" -i "$PYPI"
  # 新版 huggingface_hub 已废弃 huggingface-cli，改用 hf 命令
  hf download IDEA-Research/grounding-dino-base \
    --local-dir "$MODEL_DIR" \
    --exclude "*.bin"
fi

echo ""
echo "==> 环境准备完成。下一步："
echo "    1) 把数据集上传并解压（建议目录名用英文，如 /root/autodl-tmp/dataset）"
echo "    2) 冒烟验证："
echo "       python scripts/smoke_test.py --data-root /root/autodl-tmp/dataset"
echo "    3) 全量推理："
echo "       python scripts/infer.py --data-root /root/autodl-tmp/dataset"
echo "    4) 导出提交包："
echo "       python scripts/export_submission.py --data-root /root/autodl-tmp/dataset"
echo "    详见 docs/说明/AutoDL部署说明.md"