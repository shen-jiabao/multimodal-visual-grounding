#!/bin/bash
# 轮询 rerun3 完成标记（供本地后台反复调用；单次最多等 10 分钟）
D=/root/autodl-tmp/work/code27b/outputs/RERUN3_DONE.txt
for i in $(seq 1 10); do
  if [ -s "$D" ]; then
    echo "=== DONE ==="
    cat "$D"
    exit 0
  fi
  sleep 60
done
echo "=== STILL RUNNING ==="
pgrep -af "scripts/infer.py" | grep -v grep | head -2
tail -2 /root/autodl-tmp/work/code27b/outputs/run_rerun3.log
