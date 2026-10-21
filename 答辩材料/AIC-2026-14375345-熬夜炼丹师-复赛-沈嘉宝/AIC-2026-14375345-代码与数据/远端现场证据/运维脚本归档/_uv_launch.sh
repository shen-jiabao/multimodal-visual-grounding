#!/bin/bash
# 内联启动 uv 版建环境（不依赖远端存在 _q38_build_launch.sh）
set -u
R=/root/rivermind-data
S=$R/_q38_build.sh

if pgrep -f "_q38_build.sh" >/dev/null 2>&1; then
  echo "已在运行：$(pgrep -f _q38_build.sh | tr '\n' ' ')"; exit 0
fi

: > "$R/_q38_build.out"
setsid nohup bash "$S" > "$R/_q38_build.out" 2>&1 < /dev/null &
sleep 3
echo "已启动 pid: $(pgrep -f _q38_build.sh | tr '\n' ' ')"
echo "日志: $R/_q38_build.out  |  $R/q38_env.log"
echo "--- _q38_build.out 前 10 行 ---"
head -10 "$R/_q38_build.out" 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="
