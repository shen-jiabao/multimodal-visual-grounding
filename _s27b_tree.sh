#!/bin/bash
# 盘点 code27b 交付所需文件：代码树 + requirements + 模型清单/哈希
cd /root/autodl-tmp/work/code27b || exit 1
echo "=== 代码树（排除 outputs 大文件）==="
find . -path ./outputs -prune -o -type f -print | sort
echo "=== 体积 ==="
du -sh src scripts configs 2>/dev/null
echo "=== requirements.txt ==="
cat requirements.txt
echo "=== 模型目录 ==="
ls -la /root/autodl-tmp/models/Qwen3.8-27B/ | head -30
echo "=== 模型哈希（若有）==="
find /root/autodl-tmp/models/Qwen3.8-27B/ -name "*.json" -maxdepth 1 | head
sha256sum /root/autodl-tmp/models/Qwen3.8-27B/*.safetensors 2>/dev/null | head -3
echo "=== 实测环境 ==="
/root/miniconda3/bin/python -c "import torch,transformers;print('torch',torch.__version__);print('tf',transformers.__version__)" 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -2
