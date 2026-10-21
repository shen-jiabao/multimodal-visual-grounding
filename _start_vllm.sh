#!/bin/bash
# 按**原始命令行**恢复 vLLM 32B-AWQ 服务（取自 /proc/153/cmdline，逐字保留）。
# 用途：若探针对比后决定走 32B 路线，用本脚本把服务起回来。
cd /root
setsid nohup /root/miniconda3/envs/myconda/bin/vllm serve \
  /root/models/Qwen3-VL-32B-Instruct-AWQ \
  --served-model-name qwen3-vl-32b \
  --host 0.0.0.0 --port 8000 \
  --swap-space 4 --max-num-seqs 8 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --allowed-local-media-path /root/assets \
  > /srv/output/vllm_restart.log 2>&1 < /dev/null &
echo '已发出启动命令，等待就绪（最多 180s）...'
for i in $(seq 1 36); do
  sleep 5
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models)
  echo "  第 ${i} 次: HTTP $code"
  if [ "$code" = "200" ]; then break; fi
done
curl -s -m 8 http://127.0.0.1:8000/v1/models | head -c 200
echo
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
