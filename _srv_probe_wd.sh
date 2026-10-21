#!/usr/bin/env bash
# 新云端实例环境探测（只读）
set +e

echo "=== host ==="
hostname
cat /etc/os-release 2>/dev/null | head -n 2

echo "=== conda envs ==="
/root/miniconda3/bin/conda env list 2>/dev/null

echo "=== python / torch ==="
PY=/root/miniconda3/bin/python
$PY -c "import sys;print('python',sys.version.split()[0])"
$PY -c "import torch;print('torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())" 2>&1 | tail -n 3

echo "=== gpu ==="
ls -l /usr/bin/nvidia-smi 2>&1
/usr/bin/nvidia-smi -L 2>&1 | head -n 3
cat /proc/driver/nvidia/version 2>&1 | head -n 2

echo "=== disk ==="
df -h / /root/autodl-tmp 2>/dev/null

echo "=== /root/autodl-tmp ==="
ls -la /root/autodl-tmp 2>/dev/null

echo "=== homework/data dirs ==="
ls -la /root/autodl-tmp/rivermind-data 2>/dev/null | head -n 20

echo "=== done ==="
