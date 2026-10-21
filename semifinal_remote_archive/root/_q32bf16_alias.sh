#!/bin/bash
# 挂别名重启 bf16 服务（同一份权重，两个服务名）→ 跑接线自校验
#
# 为什么要别名：`_probe_models.py`（归档参照实现）把 API_MODEL 写死为 `qwen3-vl-32b`。
# 接线判据要求"我方管线 vs 归档实现"打**同一端点**，故给 bf16 权重挂别名，
# 而不是改归档代码（改了就失去"独立参照"的意义）。
set -u
R=/root/rivermind-data
cd $R

echo "=== 1) 停旧服务 ==="
pkill -f "_get_serve_q32bf16.sh" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo; echo "=== 2) 带别名重启 ==="
: > $R/serve32bf16f.log
Q32_UTIL=0.96 Q32_MAXLEN=8192 Q32_MAXBATCH=2048 \
Q32_MMLIMIT='{"image":1,"video":0}' \
Q32_EXTRA='--enforce-eager' Q32_ATTN=default \
Q32_ALIAS=qwen3-vl-32b \
  setsid nohup bash _get_serve_q32bf16.sh > $R/serve32bf16f.log 2>&1 < /dev/null &
echo "  已启动｜主名 qwen3-vl-32b-bf16｜别名 qwen3-vl-32b"

ready=0
for i in $(seq 1 42); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen3-vl-32b-bf16; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; ready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then echo "  ✗ 进程退出"; break; fi
done
[ "$ready" = "0" ] && { tail -12 $R/serve32bf16f.log | cut -c1-160 | sed 's/^/  /'; exit 1; }
echo "  服务端可见模型名："
curl -s --max-time 8 http://127.0.0.1:8000/v1/models | tr ',' '\n' | grep '"id"' | sed 's/^/    /'

echo; echo "=== 3) 接线自校验（两臂现在打同一端点） ==="
QWEN_ENGINE=api32b QWEN_API_MODEL=qwen3-vl-32b-bf16 QWEN_DATA=$R/dataset \
QWEN_CORES=$R/_ordinal_cores_r2.json \
  timeout 900 $R/venv/bin/python -u _smoke_qwen32b.py --mode both --n 20 2>&1 \
  | tail -32 | cut -c1-190 | sed 's/^/  /'

echo; echo "=== 4) GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="