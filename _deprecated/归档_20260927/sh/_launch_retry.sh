#!/bin/bash
# 二次枚举：逐图定制 prompt，独立输出文件，nohup 脱离 SSH 通道
set -e
LOG=/root/output/ordinal_retry.log
OUT=/root/output/locate_ordinal_retry.json
mkdir -p /root/output
if pgrep -f _locate_ordinal_batch.py >/dev/null; then
  echo "已有批处理在运行，不重复启动"
  exit 1
fi
# 输出文件若存在则归档，避免混入旧结果
[ -f "$OUT" ] && mv "$OUT" "$OUT.prev.$(date +%s)"
BATCH_MANIFEST=/root/ordinal_retry_manifest.json \
BATCH_OUT="$OUT" \
nohup /root/locate_venv/bin/python /root/_locate_ordinal_batch.py \
    > "$LOG" 2>&1 &
echo "已启动 PID=$!，日志 $LOG，输出 $OUT"
sleep 3
pgrep -af _locate_ordinal_batch.py