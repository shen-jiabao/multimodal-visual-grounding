#!/bin/bash
# bf16 起服【按 traceback 的原始证据再试一次：限 video=0】
#
# 证据（上一次 traceback，非推测）：
#   gpu_model_runner.py:3344  "Encoder cache will be initialized with a budget of 151250 tokens,
#                              and profiled with **1 video items of the max size**"
#   gpu_model_runner.py:3xxx  "expanded = output.new_zeros(" ← OOM 5.77 GiB 就发生在这里
# ⇒ 5.77 GiB 由"最大尺寸视频"的编码器探测产生；此前只限了 image，**没限 video**。
#
# 策略：本条**只试一次**（video:0 + 限制最大像素）；失败则立刻转 AWQ（其权重 20 GB 在盘上，
#   单卡 40G 此前已成功起服并跑出 0.7766），不再在 bf16 上调参。
set -u
R=/root/rivermind-data
LOG=$R/serve32bf16e.log
AWQ=$R/models/Qwen3-VL-32B-Instruct-AWQ

echo "=== 1) 清场 ==="
pkill -f "_get_serve_q32bf16.sh" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo; echo "=== 2) bf16 + video:0（唯一一次） ==="
cd $R
: > "$LOG"
Q32_UTIL=0.96 Q32_MAXLEN=8192 Q32_MAXBATCH=2048 \
Q32_MMLIMIT='{"image":1,"video":0}' \
Q32_EXTRA='--enforce-eager' Q32_ATTN=default \
  setsid nohup bash _get_serve_q32bf16.sh > "$LOG" 2>&1 < /dev/null &
echo "  已启动｜mm 限制 {\"image\":1,\"video\":0}｜日志 $LOG"

ready=0
for i in $(seq 1 36); do
  sleep 10
  if curl -s --max-time 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -q qwen3-vl-32b-bf16; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; ready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then echo "  ✗ 进程退出（$(( i * 10 )) 秒内）"; break; fi
  [ $(( i % 3 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$R/vllm32bf16.log" 2>/dev/null | cut -c1-100)"
done

if [ "$ready" = "1" ]; then
  echo; echo "=== 判定：bf16 起服成功 ==="
  grep -E "Encoder cache|KV cache|maximum concurrency|model weights took" "$R/vllm32bf16.log" \
    | tail -6 | cut -c1-170 | sed 's/^/  /'
  curl -s --max-time 10 http://127.0.0.1:8000/v1/models | cut -c1-300 | sed 's/^/  /'
  echo "  ⇒ 下一步：_smoke_qwen32b.py 接线自校验 → 20 条目检 → 全量"
  echo "=== 结束（成功分支） ==="
  exit 0
fi

echo; echo "=== 判定：bf16 仍失败 ⇒ 转 AWQ（不再调参） ==="
grep -nE "profiled with|Encoder cache|OutOfMemory|Tried to allocate|Assertion" "$R/vllm32bf16.log" \
  | tail -6 | cut -c1-170 | sed 's/^/  /'
tail -8 "$LOG" | cut -c1-150 | sed 's/^/  /'

echo; echo "=== 3) 起 AWQ 32B（1 张卡，TP=1） ==="
pkill -f "vllm serve" 2>/dev/null; sleep 8
ALOG=$R/vllm32awq2.log
: > "$ALOG"
CUDA_VISIBLE_DEVICES=0 setsid nohup $R/venv/bin/vllm serve "$AWQ" \
  --served-model-name qwen3-vl-32b \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  > "$ALOG" 2>&1 < /dev/null &
echo "  已启动｜日志 $ALOG"

aready=0
for i in $(seq 1 36); do
  sleep 10
  if curl -s --max-time 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -q qwen3-vl-32b; then
    echo "  ✓ AWQ 就绪（$(( i * 10 )) 秒）"; aready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then echo "  ✗ AWQ 进程退出"; break; fi
  [ $(( i % 3 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$ALOG" 2>/dev/null | cut -c1-100)"
done
[ "$aready" = "0" ] && { echo "  AWQ 日志尾："; tail -15 "$ALOG" | cut -c1-170 | sed 's/^/    /'; }

echo; echo "=== 4) GPU 占用 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="