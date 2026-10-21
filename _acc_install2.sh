#!/bin/bash
# A 路线：换源重试 flash-linear-attention
PY=/root/miniconda3/bin/python
echo "=== 清华源 ==="
$PY -m pip install --no-deps -q -i https://pypi.tuna.tsinghua.edu.cn/simple flash-linear-attention 2>&1 | tail -4
echo "=== import 检测 ==="
$PY -c "import flash_linear_attention as f;print('fla',getattr(f,'__version__','?'))" 2>&1 | tail -2
$PY -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule;print('gated_delta_rule OK')" 2>&1 | tail -2
echo "=== 环境未被动 ==="
$PY -c "import torch,triton;print('torch',torch.__version__,'triton',triton.__version__)"
