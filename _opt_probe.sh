#!/bin/bash
cd /root/autodl-tmp/work/code27b 2>/dev/null || { echo "NO_CODE_DIR"; exit 0; }
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== 相关进程 ==="
pgrep -af 'python|infer' | head -20
echo "=== outputs ==="
ls -la outputs/ | head -40
echo "=== predictions 文件 ==="
ls -la outputs/predictions_*.json 2>/dev/null
echo "=== configs ==="
ls -la configs/
echo "=== scripts ==="
ls -la scripts/
echo "=== src 顶层 ==="
ls src/
echo "=== code dir 顶层 ==="
ls -la
echo "=== 磁盘 ==="
df -h /root/autodl-tmp | tail -2
