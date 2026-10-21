#!/bin/bash
# 启动 Qwen3.8-27B 权重下载（ModelScope 通道 + HF sha256 校验），后台分离运行
set -u
R=/root/rivermind-data
S=$R/_q38_dl.sh

if pgrep -f "_q38_dl.sh" >/dev/null 2>&1; then
  echo "已在运行：$(pgrep -f _q38_dl.sh | tr '\n' ' ')"; exit 0
fi
if pgrep -f "Qwen3.8-27B/resolve" >/dev/null 2>&1; then
  echo "已有 curl 分片进程在跑：$(pgrep -cf 'Qwen3.8-27B/resolve') 个"; exit 0
fi

: > "$R/_q38_dl.out"
setsid nohup bash "$S" > "$R/_q38_dl.out" 2>&1 < /dev/null &
sleep 5
echo "已启动 pid: $(pgrep -f _q38_dl.sh | tr '\n' ' ')"
echo "日志: $R/_q38_dl.out  |  $R/q38_dl.log"
echo "--- _q38_dl.out 前 20 行 ---"
head -20 "$R/_q38_dl.out" 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="
