#!/bin/bash
# 235B-AWQ 下载进度（纯读）
set -u
R=/root/rivermind-data
DIR=$R/models/Qwen3-VL-235B-A22B-Instruct-AWQ
LOG=$R/dl235b.log

echo "=== 进程 ==="
if pgrep -f 'snapshot_download' >/dev/null 2>&1; then
  echo "  ✓ 仍在下载  PID=$(pgrep -f snapshot_download | head -1)"
else
  echo "  ○ 下载进程已退出"
fi

echo; echo "=== 已落盘体积（目标 116.64 GB）==="
sz=$(du -sm "$DIR" 2>/dev/null | cut -f1)
echo "  ${sz:-0} MB  ($(awk -v s="${sz:-0}" 'BEGIN{printf "%.1f%%", s/119436*100}') of 116.64 GB)"
ls -1 "$DIR"/*.safetensors 2>/dev/null | wc -l | sed 's/^/  分片已落 /;s/$/ 个（共 42）/'

echo; echo "=== 日志尾 5 行 ==="
tail -5 "$LOG" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="