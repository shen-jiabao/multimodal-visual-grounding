#!/bin/bash
# 停止 vLLM 以腾出显存（8B 探针需要 ~17GB，32B 已占 40.8GB）。
# 恢复命令见 _start_vllm.sh，两者成对使用。
echo '=== 停止 vLLM ==='
pkill -f 'vllm serve'
for i in $(seq 1 12); do
  sleep 3
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  echo "  第 ${i} 次检查: 残留 GPU 进程 $n"
  if [ "$n" = "0" ]; then break; fi
done
echo '=== GPU 现状 ==='
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
