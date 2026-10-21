#!/bin/bash
# 建隔离 overlay venv（不改动 myconda 的 transformers 5.17），仅用于 LocateAnything
set -e
VENV=/root/locate_venv
BASEPY=/root/miniconda3/envs/myconda/bin/python
if [ ! -x "$VENV/bin/python" ]; then
  echo "创建 overlay venv（--system-site-packages 复用系统 torch）..."
  $BASEPY -m venv --system-site-packages "$VENV"
fi
echo "===== 安装 transformers 4.52.4（仅装入 $VENV，不影响 myconda）====="
"$VENV/bin/pip" install -q --disable-pip-version-check "transformers==4.52.4"
echo "===== 版本自检 ====="
"$VENV/bin/python" - <<'PY'
import transformers, torch
print("overlay transformers:", transformers.__version__)
print("system torch:", torch.__version__, "cuda:", torch.cuda.is_available())
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
print("Qwen3ForCausalLM import OK")
PY
echo "===== myconda 仍是（确认未被改动）====="
$BASEPY -c "import transformers; print('myconda transformers:', transformers.__version__)"