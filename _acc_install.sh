#!/bin/bash
# A 路线：装 flash-linear-attention（--no-deps 避免动 torch/triton），再验版本
PY=/root/miniconda3/bin/python
set -x
$PY -m pip install --no-deps -q flash-linear-attention 2>&1 | tail -5
set +x
echo "=== import 检测 ==="
$PY -c "import flash_linear_attention as f;print('fla',getattr(f,'__version__','?'))" 2>&1 | tail -3
$PY -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule;print('gated_delta_rule OK')" 2>&1 | tail -3
echo "=== 环境未被动检测 ==="
$PY -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__)"
