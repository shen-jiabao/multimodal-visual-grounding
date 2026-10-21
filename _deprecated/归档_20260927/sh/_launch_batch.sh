#!/bin/bash
# 以 nohup 脱离 SSH 通道运行批量枚举，日志落文件，立即返回（轮询日志看进度）
set -e
LOG=/root/output/ordinal_batch.log
mkdir -p /root/output
# 已在跑则不重复启动
if pgrep -f _locate_ordinal_batch.py >/dev/null; then
  echo "已在运行，不重复启动"
else
  nohup /root/locate_venv/bin/python /root/_locate_ordinal_batch.py \
      > "$LOG" 2>&1 &
  echo "已启动 PID=$!，日志 $LOG"
fi
sleep 3
echo "----- 启动确认 -----"
pgrep -af _locate_ordinal_batch.py