#!/bin/bash
# 启动器：后台分离执行 G1 复核，立即退出（避免本地 SSH 读超时）
set -u
R=/root/rivermind-data
pkill -f _q38_preserve_run.sh 2>/dev/null && echo "已停旧进程" || echo "（无旧进程）"
sleep 1
cd $R
setsid nohup bash "$R/_q38_preserve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 2
echo "已启动；当前进程数 $(pgrep -cf _q38_preserve_run.sh || echo 0)"
echo "日志文件：$([ -f "$R/_q38_preserve.log" ] && echo "已建（$(wc -l < "$R/_q38_preserve.log") 行）" || echo '尚未创建')"