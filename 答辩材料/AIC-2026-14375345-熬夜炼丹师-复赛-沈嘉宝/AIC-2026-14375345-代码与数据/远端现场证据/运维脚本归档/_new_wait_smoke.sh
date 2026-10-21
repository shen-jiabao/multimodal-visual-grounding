#!/bin/bash
# 等 vLLM 就绪，随即跑接线自校验烟测（省一次往返）。
set -u
PY=/root/rivermind-data/venv/bin/python
DATA=/root/rivermind-data/dataset
API=http://127.0.0.1:8000/v1

echo "=== 等 vLLM 就绪（最多 15 分钟）==="
ready=0
for i in $(seq 1 90); do
  if curl -s --max-time 5 "$API/models" 2>/dev/null | grep -q "qwen3-vl-32b"; then
    echo "  ✓ 就绪（等待 $((i*10)) 秒）"
    curl -s "$API/models" | head -c 300; echo
    ready=1
    break
  fi
  sleep 10
  if [ $((i % 6)) -eq 0 ]; then
    echo "    …已等 $((i*10)) 秒 | 日志尾: $(tail -1 /root/rivermind-data/vllm_serve.log 2>/dev/null | cut -c1-100)"
  fi
done
if [ "$ready" -eq 0 ]; then
  echo "★ 15 分钟未就绪，服务日志尾部："
  tail -30 /root/rivermind-data/vllm_serve.log
  exit 3
fi

echo
echo "=== 跑接线自校验烟测 ==="
cd /root
QWEN_ENGINE=api32b QWEN_API_BASE="$API" QWEN_DATA="$DATA" \
  $PY -u /root/_smoke_qwen32b.py --data "$DATA" 2>&1
echo
echo "烟测退出码 $?"