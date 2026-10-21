#!/bin/bash
# 读 Qwen3.8-27B 起服日志
set -u
R=/root/rivermind-data
L=$R/_q38_serve.log
echo "进程 $(pgrep -cf _q38_serve_run.sh || echo 0)｜vllm $(pgrep -cf 'vllm serve' || echo 0)｜日志 $(wc -l < "$L" 2>/dev/null || echo 0) 行"
echo "--- 日志 ---"
cat "$L" 2>/dev/null | cut -c1-150
echo "---"
grep -q "SERVE-DONE" "$L" 2>/dev/null && echo "状态：流程结束" || echo "状态：进行中"
echo "--- vLLM 关键行 ---"
grep -E "Resolved architecture|model weights took|Encoder cache|Available KV cache|KV cache size|OutOfMemory|out of memory" "$R/vllm38.log" 2>/dev/null | tail -10 | cut -c1-160 | sed 's/^/  /'
curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 200 | sed 's/^/  /'