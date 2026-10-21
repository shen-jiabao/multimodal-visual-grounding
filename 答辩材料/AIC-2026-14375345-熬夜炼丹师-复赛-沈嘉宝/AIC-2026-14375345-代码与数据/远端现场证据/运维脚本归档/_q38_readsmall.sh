#!/bin/bash
set -u
R=/root/rivermind-data
L=$R/_q38_small.log
echo "进程 $(pgrep -cf _q38_fixsmall.sh || echo 0)｜日志 $(wc -l < "$L" 2>/dev/null || echo 0) 行"
echo "---"
cat "$L" 2>/dev/null | cut -c1-150
echo "---"
grep -q "=== DONE ===" "$L" 2>/dev/null && echo "状态：完成" || echo "状态：进行中"