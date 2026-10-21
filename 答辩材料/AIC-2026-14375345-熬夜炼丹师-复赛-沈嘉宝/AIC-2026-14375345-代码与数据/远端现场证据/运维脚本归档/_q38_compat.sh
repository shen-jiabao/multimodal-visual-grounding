#!/bin/bash
# cu130 兼容性核查 + 内核补齐尝试 + 磁盘账
set -u
R=/root/rivermind-data
V=$R/venv_q38

echo "=== ① 驱动版本 vs CUDA 13.0 要求 ==="
nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
echo "  （CUDA 13.0 一般要求驱动 ≥ 580；CUDA 12.8 要求 ≥ 570）"
echo "  nvidia-smi 报的 CUDA 版本："
nvidia-smi 2>/dev/null | grep -i "CUDA Version" | sed 's/^/    /'
echo "  venv(旧) torch：$($R/venv/bin/python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "  venv_q38 torch：$($V/bin/python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "  ★说明：无卡时 torch.cuda.is_available() 必为 False；驱动兼容性要看上面驱动版本号"

echo; echo "=== ② 内核补齐尝试（纯 python wheel） ==="
$V/bin/python -m pip install -q flash-linear-attention 2>&1 | tail -3 | sed 's/^/  /'
$V/bin/python -c "import flash_linear_attention as m;print('  ✓ flash_linear_attention', getattr(m,'__version__','ok'))" 2>&1 | sed 's/^/  /'
echo "  --- causal-conv1d（本想用 pip 版 nvcc 编译，先只试 wheel）---"
$V/bin/python -m pip install -q --only-binary=:all: causal-conv1d 2>&1 | tail -2 | sed 's/^/  /'
$V/bin/python -c "import causal_conv1d as m;print('  ✓ causal_conv1d', getattr(m,'__version__','ok'))" 2>&1 | sed 's/^/  /'

echo; echo "=== ③ vLLM 是否自带 GDN 实现（决定是否需要上面两个内核） ==="
ls $V/lib/python3.10/site-packages/vllm/model_executor/models/ 2>/dev/null | grep -iE "qwen3_5|gdn|delta" | head -6 | sed 's/^/  /'
grep -rlE "gated.?delta|GatedDelta" $V/lib/python3.10/site-packages/vllm/ 2>/dev/null | head -5 | sed 's/^/  引用: /'

echo; echo "=== ④ 磁盘账 ==="
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'
echo "  当前可用 $(df -BG --output=avail $R | tail -1 | tr -dc '0-9') GB"
echo "  删 AWQ(20G) 后 ≈ $(($(df -BG --output=avail $R | tail -1 | tr -dc '0-9') + 20)) GB；模型需 55.6 GB"
echo "=== 结束 ==="