#!/bin/bash
# 序数层批量枚举（overlay venv，启动前查 GPU，脚本内逐图复查/续跑）
set +e
echo "启动时间 $(date '+%H:%M:%S')"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
ps -ef | grep -v grep | grep -E "test_visualize|locate_run" | head
echo "----- 运行 -----"
/root/locate_venv/bin/python /root/_locate_ordinal_batch.py
echo "返回码 $?"