#!/bin/bash
# 在远端运行 Locate 多候选探针（仅图 000023，约 1 分钟）
set -u
echo "===== GPU 占用（应空闲）====="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || true
echo "===== 现存 python 进程（确认无其他推理在跑）====="
ps -ef | grep -E "locate_run|python" | grep -v grep | head
PY=/root/miniconda3/envs/myconda/bin/python
[ -x "$PY" ] || PY=python3
echo "使用解释器: $PY"
$PY /root/_locate_multi_probe.py