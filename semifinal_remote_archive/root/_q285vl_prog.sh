#!/bin/bash
# Qwen2.5-VL-72B-AWQ 下载进度
set -u
R=/root/rivermind-data
D=$R/models/Qwen2.5-VL-72B-Instruct-AWQ
LOG=$R/q285vl_dl.log
TOT=41028

echo "=== 进程 ==="
echo "  monitor $(pgrep -cf _q285_monitor.sh || echo 0)｜worker $(pgrep -cf _q285_w.sh || echo 0)"
echo "  curl   $(pgrep -cf "Qwen2.5-VL-72B-Instruct-AWQ/resolve" || echo 0)"

echo; echo "=== 目录 ==="
sz=$(( $(du -sm "$D" 2>/dev/null | cut -f1 || echo 0) ))
echo "  ${sz} MB / ${TOT} MB（$(( sz * 100 / TOT ))%）"
ls -1 "$D"/*.safetensors 2>/dev/null | wc -l | sed 's/^/  已合并分片 /'
echo "  分片残块 $(ls -1 "$D/.parts" 2>/dev/null | wc -l) 个"

echo; echo "=== 日志尾 10 行 ==="
tail -10 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/  /'

echo; echo "=== 磁盘 / GPU ==="
df -h $R | tail -1 | sed 's/^/  /'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  GPU /'
echo "=== 结束 ==="
