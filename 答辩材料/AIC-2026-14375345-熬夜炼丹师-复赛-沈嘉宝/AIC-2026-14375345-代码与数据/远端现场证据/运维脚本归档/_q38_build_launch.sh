#!/bin/bash
# 把 _q38_build.sh 分离执行（建环境中途无输出，同步跑会撞本地 SSH 读超时）。
#   前置：需先用 `_rc.py --put` 把 _q38_build.sh 放到 $R。
set -u
R=/root/rivermind-data
S=$R/_q38_build.sh

[ -f "$S" ] || { echo "★ 缺 $S（请先 --put _q38_build.sh）"; exit 9; }

if pgrep -f "_q38_build.sh" >/dev/null 2>&1; then
  echo "已在运行：$(pgrep -f _q38_build.sh | tr '\n' ' ')"
  exit 0
fi

# 先清掉上一轮的 venv 日志头，避免读旧内容误判
: > "$R/_q38_build.out"
setsid nohup bash "$S" > "$R/_q38_build.out" 2>&1 < /dev/null &
sleep 5

echo "已启动 pid: $(pgrep -f _q38_build.sh | tr '\n' ' ')"
echo "日志: $R/_q38_build.out  |  $R/q38_env.log"
echo "--- _q38_build.out 前 8 行 ---"
head -8 "$R/_q38_build.out" 2>/dev/null | sed 's/^/  /'
echo "--- 磁盘 ---"
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
