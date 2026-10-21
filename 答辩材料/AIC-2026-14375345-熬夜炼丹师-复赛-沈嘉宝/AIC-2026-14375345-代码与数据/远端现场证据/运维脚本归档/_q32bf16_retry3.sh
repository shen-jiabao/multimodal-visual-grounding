#!/bin/bash
# bf16 起服【最后一次尝试：V0 引擎】
#
# 依据：三次失败点完全相同、5.77 GiB 恒定不变 ⇒ 该分配由模型结构决定，
#   不受 batch token / maxlen / mm 限制影响。V0 引擎用解析式估显存，不做 dummy 前向，
#   可能绕过它。**只试这一次**；再失败即判定 2×40G 装不下 bf16，转 AWQ 路线。
set -u
R=/root/rivermind-data
LOG=$R/serve32bf16d.log

echo "=== 0) 先把上次 OOM 的 traceback 存档（证据，不猜） ==="
sed -n '70,100p' "$R/vllm32bf16.log" 2>/dev/null | cut -c1-190 | sed 's/^/  /'

echo; echo "=== 1) 清场 ==="
pkill -f "_get_serve_q32bf16.sh" 2>/dev/null; pkill -f "vllm serve" 2>/dev/null; sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo; echo "=== 2) 最后一次尝试：VLLM_USE_V1=0（保守配置不变） ==="
cd $R
: > "$LOG"
Q32_UTIL=0.96 Q32_MAXLEN=8192 Q32_MAXBATCH=2048 Q32_MMLIMIT='{"image":1}' \
Q32_EXTRA="--enforce-eager" Q32_ATTN=default VLLM_USE_V1=0 \
  setsid nohup bash _get_serve_q32bf16.sh > "$LOG" 2>&1 < /dev/null &
echo "  已启动｜VLLM_USE_V1=0｜日志 $LOG"

echo; echo "=== 3) 观察（最多 7 分钟） ==="
ready=0
for i in $(seq 1 42); do
  sleep 10
  if curl -s --max-time 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -q qwen3-vl-32b-bf16; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; ready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "  ✗ 进程退出（$(( i * 10 )) 秒内）"; break
  fi
  [ $(( i % 3 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$R/vllm32bf16.log" 2>/dev/null | cut -c1-100)"
done

echo; echo "=== 4) 判定 ==="
if [ "$ready" = "1" ]; then
  echo "  ★ V0 引擎起服成功 ⇒ bf16 路线可继续"
  curl -s --max-time 10 http://127.0.0.1:8000/v1/models | cut -c1-320 | sed 's/^/    /'
  grep -E "KV cache|maximum concurrency|model weights took" "$R/vllm32bf16.log" | tail -5 | cut -c1-170 | sed 's/^/    /'
else
  echo "  ★ V0 亦失败 ⇒ **判定：bf16 32B 在 2×40G 上不可行**"
  echo "  报错行："
  grep -nE "OutOfMemory|out of memory|No module|not supported|Error" "$R/vllm32bf16.log" 2>/dev/null \
    | tail -8 | cut -c1-190 | sed 's/^/    /'
  echo "  起服脚本输出尾："
  tail -12 "$LOG" | cut -c1-170 | sed 's/^/    /'
fi
echo; echo "=== 5) GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="