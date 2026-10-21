#!/bin/bash
# 清理：卸载残缺的 flash-linear-attention（缺 fla/ops，但 transformers 会误判可用）
PY=/root/miniconda3/bin/python
$PY -m pip uninstall -y flash-linear-attention 2>&1 | tail -3
echo "=== 确认 ==="
$PY -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f;print('flash_linear_attention_available',f())" 2>&1 | tail -2
$PY -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__)"
