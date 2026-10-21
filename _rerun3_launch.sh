#!/bin/bash
# 第三次全量重跑（同码同配置，纯解码随机性）：infer 后台 + 完成监视器
set -e
cd /root/autodl-tmp/work/code27b || exit 1
mkdir -p outputs
[ -s outputs/predictions_rerun3.json ] && { echo "RERUN3 已存在，勿重复发车"; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

nohup /root/miniconda3/bin/python scripts/infer.py \
  --config configs/server_32b.yaml \
  --output outputs/predictions_rerun3.json \
  > outputs/run_rerun3.log 2>&1 &
PID=$!
echo "INFER_PID=$PID"
sleep 45
echo "=== 45 秒后进程 ==="
kill -0 "$PID" && echo "alive" || { echo "DEAD"; tail -20 outputs/run_rerun3.log; exit 1; }
echo "=== 发车自检：日志须含 1295 张/5690 条 与 multi_box/thinking=False ==="
head -10 outputs/run_rerun3.log | grep -E "1295|5690|multi_box" || echo "WARN: 自检行未出现（可能还在加载权重）"
nohup bash /root/_rerun3_watch.sh "$PID" > /dev/null 2>&1 &
echo "WATCH_PID=$!"
tail -3 outputs/run_rerun3.log
