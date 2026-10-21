#!/bin/bash
# 全量 GD 三模态推理：输出到独立新文件，不触碰已锁定结果
set -e
cd /root/infer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false CUDA_DEVICE_ORDER=PCI_BUS_ID
if pgrep -f predict_gd_3modal.py >/dev/null; then
  echo "已有 predict_gd_3modal.py 在跑，不重复启动"
  exit 1
fi
setsid nohup nice -n 10 /root/miniconda3/envs/myconda/bin/python -u predict_gd_3modal.py \
    --out /root/infer/result_gd_3modal_full.json \
    > /root/infer/run_full_new.log 2>&1 < /dev/null &
sleep 4
echo "PID=$(pgrep -f predict_gd_3modal.py | head -1)"
echo "日志 /root/infer/run_full_new.log"
echo "输出 /root/infer/result_gd_3modal_full.json"
pgrep -af predict_gd_3modal.py