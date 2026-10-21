#!/bin/bash
cd /root/autodl-tmp/work/code27b
echo "===== configs/server_32b.yaml ====="
cat configs/server_32b.yaml
echo "===== scripts/infer.py ====="
cat scripts/infer.py
echo "===== run_annotate.log 头 25 行 ====="
head -25 outputs/run_annotate.log
echo "===== run_annotate.log 尾 15 行 ====="
tail -15 outputs/run_annotate.log
echo "===== run_rerun3.log 全文 ====="
cat outputs/run_rerun3.log
echo "===== README 关键段（前 120 行） ====="
head -120 README.md
