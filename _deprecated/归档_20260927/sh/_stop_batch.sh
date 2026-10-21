#!/bin/bash
# 停止当前卡住的批处理进程（仅匹配我们自己的脚本），释放其显存
set +e
PIDS=$(pgrep -f _locate_ordinal_batch.py)
if [ -n "$PIDS" ]; then
  echo "kill: $PIDS"
  kill $PIDS
  sleep 5
  REMAIN=$(pgrep -f _locate_ordinal_batch.py)
  [ -n "$REMAIN" ] && { echo "残留 $REMAIN，强制 kill"; kill -9 $REMAIN; sleep 3; }
else
  echo "无运行中的批处理进程"
fi
echo "----- 停止后 GPU -----"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader