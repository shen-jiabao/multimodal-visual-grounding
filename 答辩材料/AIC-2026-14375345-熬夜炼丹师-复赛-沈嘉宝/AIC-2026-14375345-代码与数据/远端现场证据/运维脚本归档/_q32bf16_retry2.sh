#!/bin/bash
# bf16 32B 起服【保守配置 v2】
#
# 实测根因：失败点在 vLLM 的 determine_available_memory() 探测跑（gpu_worker.py:263），
#   它按 max_num_batched_tokens 造 dummy 激活，默认一次要 5.77 GiB；而 2×40G 上
#   权重已占 31.07 GiB/卡（容量 39.49）⇒ 顶爆。故：
#     --max-num-batched-tokens 2048   （砍批激活；探测跑随之变小）
#     --limit-mm-per-prompt {"image":1}（限每提示 1 图；管线 QWEN_MODALITIES 缺省即 1）
#     --enforce-eager                  （省 CUDA graph 显存）
#     --max-model-len 8192             （省 KV 预分配）
#   util 提到 0.96 把"上限"抬高（注意：util 是**上限**，不是占用；真正的瓶颈是探测跑的峰值）
set -u
R=/root/rivermind-data
LOG=$R/serve32bf16c.log

echo "=== 清场 ==="
pkill -f "_get_serve_q32bf16.sh" 2>/dev/null && echo "  已停旧起服脚本" || echo "  （无）"
pkill -f "vllm serve" 2>/dev/null && echo "  已停旧 vllm" || echo "  （无）"
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/  /'

echo; echo "=== 启动（保守 v2） ==="
cd $R
: > "$LOG"
Q32_UTIL=0.96 \
Q32_MAXLEN=8192 \
Q32_MAXBATCH=2048 \
Q32_MMLIMIT='{"image":1}' \
Q32_EXTRA="--enforce-eager" \
Q32_ATTN=default \
  setsid nohup bash _get_serve_q32bf16.sh > "$LOG" 2>&1 < /dev/null &
echo "  已启动｜日志 $LOG"

echo; echo "=== 观察（最多 9 分钟） ==="
ready=0
for i in $(seq 1 54); do
  sleep 10
  if curl -s --max-time 5 "http://127.0.0.1:8000/v1/models" 2>/dev/null | grep -q qwen3-vl-32b-bf16; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; ready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "  ✗ 进程退出（$(( i * 10 )) 秒内）"; break
  fi
  [ $(( i % 3 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$R/vllm32bf16.log" 2>/dev/null | cut -c1-105)"
done

echo; echo "=== 起服脚本输出 ==="
tail -30 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/  /'

echo; echo "=== vLLM 日志关键行 ==="
grep -E "determine_available_memory|KV cache|model weights took|Resolved architecture|maximum concurrency|Available KV|GPU KV" \
  "$R/vllm32bf16.log" 2>/dev/null | tail -10 | cut -c1-170 | sed 's/^/  /'

echo; echo "=== 真实报错（若有） ==="
grep -nE "OutOfMemory|out of memory|Error|error" "$R/vllm32bf16.log" 2>/dev/null | tail -6 | cut -c1-180 | sed 's/^/  /'

echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'
[ "$ready" = "1" ] && { echo; echo "=== 服务端自证 ==="; curl -s --max-time 10 http://127.0.0.1:8000/v1/models | cut -c1-400 | sed 's/^/  /'; }
echo "=== 结束 ==="