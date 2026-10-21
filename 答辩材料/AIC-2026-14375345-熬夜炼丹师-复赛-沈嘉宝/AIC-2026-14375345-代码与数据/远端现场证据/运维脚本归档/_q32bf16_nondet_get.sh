#!/bin/bash
# 等非确定性实验完成并取回结果（分离执行后单独取，避免 SSH 拉长输出超时）
set -u
R=/root/rivermind-data
LOG=$R/_nondet.log

for i in $(seq 1 60); do
  if grep -q "摘要：" "$LOG" 2>/dev/null; then
    echo "=== 实验完成（等待 $(( i * 10 )) 秒） ==="
    break
  fi
  if ! pgrep -f "_q32bf16_nondet.py" >/dev/null; then
    echo "=== 进程已退出（$(( i * 10 )) 秒内） ==="
    break
  fi
  sleep 10
done

echo; echo "=== 完整结果 ==="
tail -30 "$LOG" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 摘要行 ==="
grep -E "摘要：|⇒|✓ 自比|★" "$LOG" 2>/dev/null | tail -12 | sed 's/^/  /'

echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="