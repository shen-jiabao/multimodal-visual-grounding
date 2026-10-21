#!/bin/bash
# 看起服进度（加载 52 GB 权重需要几分钟）
set -u
R=/root/rivermind-data

echo "=== 进程 ==="
pgrep -af "_serve_run.sh|vllm serve" 2>/dev/null | sed 's/^/  /' || echo "  （无）"

echo
echo "=== _q38_serve.log 末 26 行 ==="
tail -26 "$R/_q38_serve.log" 2>/dev/null | cut -c1-190 | sed 's/^/  /'

echo
echo "=== vllm38.log 末 10 行 ==="
tail -10 "$R/vllm38.log" 2>/dev/null | cut -c1-190 | sed 's/^/  /'

echo
echo "=== 服务探活（/v1/models）==="
out=$(curl -s --max-time 6 http://127.0.0.1:8000/v1/models 2>/dev/null)
if echo "$out" | grep -q qwen38-27b; then
  echo "  ✓ 已就绪"; echo "$out" | tr ',' '\n' | grep -E '"id"|root' | sed 's/^/    /'
else
  echo "  （尚未就绪）$(echo "$out" | head -c 120)"
fi

echo
echo "=== 显存 / 磁盘 ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
