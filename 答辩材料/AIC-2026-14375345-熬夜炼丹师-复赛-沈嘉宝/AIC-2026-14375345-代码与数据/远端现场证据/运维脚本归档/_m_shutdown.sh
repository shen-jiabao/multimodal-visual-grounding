#!/bin/bash
# 停 vLLM 释放显存 + 产物盘点（切无卡模式后所有文件仍保留）
set -u
R=/root/rivermind-data

echo "=== ① 停 vLLM ==="
pkill -f "_serve_run.sh" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
sleep 5
pkill -9 -f "vllm serve" 2>/dev/null
sleep 2
if pgrep -af "vllm serve|_serve_run.sh" >/dev/null 2>&1; then
  pgrep -af "vllm serve|_serve_run.sh" | cut -c1-100 | sed 's/^/  ★残留 /'
else
  echo "  ✓ 已停"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/  显存 /'

echo
echo "=== ② M 件产物盘点 ==="
for f in output/m_direct.json output/m_direct_filled.json output/m_retry.json; do
  [ -f "$R/$f" ] && printf "  %-40s %10d B\n" "$f" "$(stat -c%s "$R/$f")"
done
echo "  --- 回补与日志 ---"
ls -l "$R/_m_retry_qids.json" "$R/m_run.log" "$R/m_retry.log" 2>/dev/null \
  | awk '{printf "  %-40s %10d B\n", $NF, $5}'

echo
echo "=== ③ 磁盘（切无卡后不变）==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
