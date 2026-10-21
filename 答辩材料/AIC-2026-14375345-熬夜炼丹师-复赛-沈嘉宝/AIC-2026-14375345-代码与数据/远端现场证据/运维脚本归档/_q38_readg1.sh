#!/bin/bash
# 读 G1 复核结果（短命令，不阻塞）
set -u
R=/root/rivermind-data
L=$R/_q38_preserve.log
echo "进程数 $(pgrep -cf _q38_preserve_run.sh || echo 0)｜日志 $(wc -l < "$L" 2>/dev/null || echo 0) 行"
echo "---"
cat "$L" 2>/dev/null | cut -c1-140
echo "---"
grep -q "=== DONE ===" "$L" 2>/dev/null && echo "状态：完成" || echo "状态：仍在跑"