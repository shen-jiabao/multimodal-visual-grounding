#!/bin/bash
set +e
echo "时间 $(date '+%H:%M:%S')"
ps -eo pid,etime,time,%cpu,%mem,cmd | grep -v grep | grep "_locate_ordinal_batch"
echo "----- GPU -----"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "----- 该进程 GPU 利用率采样(2次,间隔3s) -----"
nvidia-smi pmon -c 2 -d 3 2>/dev/null | head -20