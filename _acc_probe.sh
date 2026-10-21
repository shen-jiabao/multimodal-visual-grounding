#!/bin/bash
# A 路线：探测加速库安装前置条件（不改环境）
PY=/root/miniconda3/bin/python
echo "=== versions ==="
$PY -c "import torch,sys;print('python',sys.version.split()[0]);print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"
echo "=== nvcc ==="
which nvcc && nvcc --version | tail -2
echo "=== 已装检测 ==="
$PY -c "import triton;print('triton',triton.__version__)" 2>&1 | tail -1
$PY -c "import flash_linear_attention as f;print('fla',getattr(f,'__version__','?'))" 2>&1 | tail -1
$PY -c "import causal_conv1d;print('causal_conv1d ok')" 2>&1 | tail -1
echo "=== pip 源 ==="
$PY -m pip config list 2>/dev/null
echo "=== torch 关键依赖 ==="
$PY -m pip list 2>/dev/null | grep -iE 'triton|torch|transformers|flash|ninja|packaging|einops' 
echo "=== 模型目录 ==="
ls -la /root/autodl-tmp/models/
echo "=== 模型 config 架构 ==="
cat /root/autodl-tmp/models/Qwen3.8-27B/config.json 2>/dev/null | head -60
