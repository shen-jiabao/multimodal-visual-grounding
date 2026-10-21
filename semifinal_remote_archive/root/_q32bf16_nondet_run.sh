#!/bin/bash
# 非确定性实验：远端分离执行 + 结果落文件（避免 SSH 拉长输出超时丢结果）
set -u
R=/root/rivermind-data
LOG=$R/_nondet.log
PY=$R/venv/bin/python

echo "=== 当前状态 ==="
echo "  nondet 进程: $(pgrep -cf '_q32bf16_nondet.py' || true) 个"
echo "  vllm serve : $(pgrep -cf 'vllm serve' || true) 个"
pgrep -af "vllm serve" | head -1 | cut -c1-120 | sed 's/^/    /'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "  日志文件: $([ -f "$LOG" ] && echo "$(wc -l < "$LOG") 行" || echo '不存在')"

if pgrep -f "_q32bf16_nondet.py" >/dev/null; then
  echo; echo "=== 实验仍在跑，日志尾 15 行 ==="
  tail -15 "$LOG" 2>/dev/null | sed 's/^/  /'
  echo "=== 结束（等待中） ==="
  exit 0
fi

if grep -q "摘要：" "$LOG" 2>/dev/null; then
  echo; echo "=== 实验已完成，结果 ==="
  tail -25 "$LOG" | sed 's/^/  /'
  echo "=== 结束（已完成） ==="
  exit 0
fi

echo; echo "=== 启动实验（分离，落文件） ==="
cd $R
: > "$LOG"
QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
  setsid nohup $PY -u _q32bf16_nondet.py > "$LOG" 2>&1 < /dev/null &
echo "  已启动 PID=$(pgrep -f _q32bf16_nondet.py | head -1)"

sleep 90
echo; echo "=== 90 秒后 ==="
tail -12 "$LOG" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="