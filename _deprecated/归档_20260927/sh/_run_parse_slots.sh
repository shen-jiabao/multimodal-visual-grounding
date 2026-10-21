#!/bin/bash
# 运行 Qwen 槽位解析 50 条（分层采样）
set -u
cd /root
PY=/root/miniconda3/envs/myconda/bin/python
[ -x "$PY" ] || PY=python3
echo "使用 $PY"
$PY /root/_parse_slots.py 50