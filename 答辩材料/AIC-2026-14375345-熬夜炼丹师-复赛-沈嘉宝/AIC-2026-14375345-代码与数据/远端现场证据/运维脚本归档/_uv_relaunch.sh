#!/bin/bash
# 核验远端脚本已更新 → 停旧构建 → 用 uv 版重启
set -u
R=/root/rivermind-data

echo "=== ① 核验远端 _q38_build.sh 已是 uv 版 ==="
if grep -q "uv pip install\|--default-index" "$R/_q38_build.sh"; then
  echo "  ✓ 已是 uv 版"
  grep -n "UVB=\|--default-index" "$R/_q38_build.sh" | sed 's/^/    /'
else
  echo "  ★ 仍是旧版！中止"; exit 1
fi

echo
echo "=== ② 停掉旧构建 ==="
pkill -f "_q38_build.sh" 2>/dev/null
pkill -f "pip install vllm" 2>/dev/null
sleep 3
if pgrep -af "_q38_build.sh|pip install vllm" >/dev/null 2>&1; then
  echo "  ★ 残留："; pgrep -af "_q38_build.sh|pip install vllm" | sed 's/^/    /'
else
  echo "  ✓ 已停止"
fi

echo
echo "=== ③ 用 uv 版重启（后台）==="
bash "$R/_q38_build_launch.sh"
echo "=== DONE ==="
