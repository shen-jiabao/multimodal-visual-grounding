#!/bin/bash
# 服务端轮询等待 GPU 空闲，再用 overlay venv 跑 Locate 多候选探针
set +e
THRESH=12000          # 需要的空闲显存 MiB（3B bf16 权重+KV+余量）
MAX_TRY=90            # 最多等 90*20s = 30 分钟
SLEEP=20

echo "开始等待 GPU 空闲（阈值 ${THRESH} MiB），当前时间 $(date +%H:%M:%S)"
ok=0
for i in $(seq 1 $MAX_TRY); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  echo "[$(date +%H:%M:%S)] try=$i free=${free}MiB used=${used}MiB gpu_procs=$nproc"
  if [ "${free:-0}" -ge "$THRESH" ]; then
    echo "GPU 已空闲（free=${free}MiB），启动探针。"
    ok=1
    break
  fi
  sleep $SLEEP
done

if [ "$ok" != "1" ]; then
  echo "TIMEOUT: 等待 30 分钟后 GPU 仍未空闲，放弃本次探针（未运行任何推理）。"
  exit 3
fi

echo "===== 启动前最终 GPU 状态 ====="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "===== 运行多候选探针（overlay transformers 4.52.4）====="
/root/locate_venv/bin/python /root/_locate_multi_probe.py
echo "===== 探针返回码 $? ====="