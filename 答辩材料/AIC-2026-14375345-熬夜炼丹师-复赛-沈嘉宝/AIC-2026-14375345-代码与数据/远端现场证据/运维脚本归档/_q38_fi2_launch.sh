#!/bin/bash
# 启动器：分离执行 _q38_fi2.sh 并落日志，立即退出
set -u
R=/root/rivermind-data
pkill -f _q38_fi2.sh 2>/dev/null && echo "已停旧进程" || echo "（无旧进程）"
sleep 1
cd "$R"
: > "$R/_q38_fi2.log"
setsid nohup bash "$R/_q38_fi2.sh" >> "$R/_q38_fi2.log" 2>&1 < /dev/null &
sleep 2
echo "已启动；进程 $(pgrep -cf _q38_fi2.sh || echo 0)"
echo "日志 $R/_q38_fi2.log"