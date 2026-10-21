#!/bin/bash
# 起服 + 闸门 5（两臂同模型名 ⇒ 请求体逐字节相同）
#
# 依据（静态证明，_q32bf16_payload_equiv.py）：
#   两条客户端路径的请求体**只差 model 字段**（我方 qwen3-vl-32b-bf16 vs 归档 qwen3-vl-32b），
#   其余（图像哈希、提示词、max_tokens、temperature）逐字节相同。
#   二者本就是我们刻意挂的**别名、指向同一份 bf16 权重**。
# ⇒ 让我方管线也走该别名，两臂请求体即完全相同，闸门 5 可在**其自身口径**下通过，
#   无需把判据放宽、也无需挑样本量。
set -u
R=/root/rivermind-data
ALIAS=qwen3-vl-32b
cd $R

echo "=== 0) GPU 实况 ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | sed 's/^/  /'
ls /dev/nvidia[0-9]* >/dev/null 2>&1 && echo "  设备节点 ✓" || { echo "  ★无设备节点，终止"; exit 9; }

echo; echo "=== 1) 起服（bf16，主名+别名；video:0） ==="
pkill -f "_get_serve_q32bf16.sh" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; sleep 6
: > $R/serve32bf16g.log
Q32_UTIL=0.96 Q32_MAXLEN=8192 Q32_MAXBATCH=2048 \
Q32_MMLIMIT='{"image":1,"video":0}' \
Q32_EXTRA='--enforce-eager' Q32_ATTN=default \
Q32_ALIAS=$ALIAS \
  setsid nohup bash _get_serve_q32bf16.sh > $R/serve32bf16g.log 2>&1 < /dev/null &
echo "  已启动（主名 qwen3-vl-32b-bf16，别名 $ALIAS）"

ready=0
for i in $(seq 1 48); do
  sleep 10
  if curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q "$ALIAS"; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; ready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then echo "  ✗ 进程退出"; break; fi
  [ $(( i % 3 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 $R/vllm32bf16.log 2>/dev/null | cut -c1-95)"
done
if [ "$ready" = "0" ]; then
  echo "  起服失败，日志尾："; tail -15 $R/serve32bf16g.log | cut -c1-160 | sed 's/^/    /'; exit 1
fi
echo "  服务端模型名与 root："
curl -s --max-time 8 http://127.0.0.1:8000/v1/models | tr ',' '\n' | grep -E '"id"|models/Qwen' | sed 's/^/    /'
echo "  权重路径（必须含 32B-Instruct 且不含 AWQ）："
grep -m1 -o "models/Qwen3-VL-[A-Za-z0-9-]*" $R/vllm32bf16.log | sed 's/^/    /'
grep -E "Encoder cache|Available KV cache" $R/vllm32bf16.log | tail -3 | cut -c1-150 | sed 's/^/    /'

echo; echo "=== 2) 闸门 5：两臂同模型名（$ALIAS），全部 30 条 ==="
QWEN_ENGINE=api32b QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_API_MODEL=$ALIAS QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
QWEN_CONC=1 \
  timeout 900 $R/venv/bin/python -u $R/_smoke_qwen32b.py --mode both --data $R/dataset 2>&1 \
  | tail -22 | cut -c1-180 | sed 's/^/  /'
echo "  烟测退出码 ${PIPESTATUS[0]}"

echo; echo "=== 3) 逆向对照：我方用 bf16 名、归档用别名（复现此前 4 条偏差的条件） ==="
QWEN_ENGINE=api32b QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_API_MODEL=qwen3-vl-32b-bf16 QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
QWEN_CONC=1 \
  timeout 900 $R/venv/bin/python -u $R/_smoke_qwen32b.py --mode both --data $R/dataset 2>&1 \
  | grep -E "逐位相同|★ 接线|IoU=" | head -8 | sed 's/^/  /'

echo; echo "=== 4) GPU 占用 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="