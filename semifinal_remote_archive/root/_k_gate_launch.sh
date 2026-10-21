#!/bin/bash
# 分离发起 K 件闸门（避免 SSH 长输出读超时）；输出落日志，另用短命令读
set -u
R=/root/rivermind-data
pkill -f "_k_gate.sh" 2>/dev/null
sleep 1
rm -f "$R/_k_gate.log"
setsid nohup bash "$R/_k_gate.sh" > "$R/_k_gate.log" 2>&1 < /dev/null &
sleep 5
echo "已发起 PID=$(pgrep -f _k_gate.sh | head -1)｜日志 $R/_k_gate.log"
echo "=== 首 12 行 ==="
head -n 12 "$R/_k_gate.log" 2>/dev/null | sed 's/^/  /'
echo "=== END ==="
