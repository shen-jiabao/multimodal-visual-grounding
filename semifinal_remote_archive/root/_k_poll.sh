#!/bin/bash
# K 件起服就绪轮询（以 HTTP 为准——避免把"前置段尚未 fork 出子进程"误判为崩溃）
#   用法：bash _k_poll.sh          （单次探测，退出码 0 = 就绪）
set -u
R=/root/rivermind-data
API=http://127.0.0.1:8000/v1

echo "===== 进程 ====="
pgrep -af "_q38_serve_run.sh|vllm serve" | cut -c1-160 || echo "  (无 vllm 进程)"
echo "===== HTTP ====="
SERVED=$(curl -s --max-time 8 "$API/models" || echo "")
if [ -z "$SERVED" ]; then
  echo "  服务未就绪（/v1/models 无响应）"
else
  echo "$SERVED" | tr ',' '\n' | grep '"id"' | sed 's/^/  /'
  echo "$SERVED" | grep -o "models/[A-Za-z0-9._-]*" | sort -u | sed 's/^/  root=/'
fi
echo "===== 日志尾 30 行 ====="
tail -n 30 "$R/vllm_k.log" 2>/dev/null | cut -c1-170 | sed 's/^/  /' \
  || echo "  (无 $R/vllm_k.log)"
echo "===== 发起器日志尾 15 行 ====="
tail -n 15 "$R/_serve_k.log" 2>/dev/null | cut -c1-170 | sed 's/^/  /'
echo "=== END ==="
[ -n "$SERVED" ] && exit 0 || exit 1
