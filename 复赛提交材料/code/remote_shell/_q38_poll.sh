#!/bin/bash
# 等 Qwen3.8-27B 服务就绪（只看 vllm 进程，不误判）
#
# 教训：旧观察循环用 `pgrep -f "vllm serve"` 判活，起服后 10 秒就查，
# 而 `_q38_serve_run.sh` 前置段（GPU 检查/权重校验/import vllm）要十几秒，
# 那会儿 vllm 还没 fork 出来 ⇒ 假报"进程退出"。本脚本改为**以 HTTP 为准**，
# 进程只在"日志已出现后端尝试"时才作为失效判据。
set -u
R=/root/rivermind-data
PORT=8000
NAME=qwen38-27b
LOG=$R/vllm38.log

echo "=== 起服状态 ==="
echo "  启动器 $(pgrep -cf _q38_serve_run.sh || echo 0) 个｜vllm $(pgrep -cf 'vllm serve' || echo 0) 个"
echo "  serve 日志 $(wc -l < "$R/_q38_serve.log" 2>/dev/null || echo 0) 行｜vllm 日志 $(wc -l < "$LOG" 2>/dev/null || echo 0) 行"

echo; echo "=== 轮询 /v1/models（最多 15 分钟） ==="
for i in $(seq 1 90); do
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$NAME"; then
    echo "  ✓ **就绪**（约 $(( i * 10 )) 秒）"
    break
  fi
  if [ $(( i % 6 )) -eq 0 ]; then
    echo "    …$(( i * 10 ))s｜$(tail -1 "$LOG" 2>/dev/null | cut -c1-100)"
  fi
  sleep 10
done

echo; echo "=== 结果 ==="
curl -s --max-time 8 "http://127.0.0.1:$PORT/v1/models" | head -c 300 | sed 's/^/  /'
echo
echo "--- vLLM 关键行 ---"
grep -E "Resolved architecture|model weights took|Encoder cache|Available KV cache|KV cache size|Starting vLLM API|Application startup complete" "$LOG" 2>/dev/null | tail -8 | cut -c1-150 | sed 's/^/  /'
echo "--- 报错行（若有） ---"
grep -iE "error|out of memory|not supported|ImportError|ModuleNotFound|unrecognized|cannot find -l|Ninja build failed" "$LOG" 2>/dev/null | tail -8 | cut -c1-170 | sed 's/^/  ✗ /'
echo "=== DONE ==="
