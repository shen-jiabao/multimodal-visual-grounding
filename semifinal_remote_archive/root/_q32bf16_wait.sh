#!/bin/bash
# 等服务就绪 + 自证（纯 bash，避免 PowerShell 破坏内联命令）
PORT=8000
NAME=qwen3-vl-32b-bf16
LOG=/root/rivermind-data/serve32bf16.log

echo "=== 等待就绪（最多 6 分钟） ==="
ready=0
for i in $(seq 1 36); do
  if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$NAME"; then
    echo "  ✓ 就绪（$(( i * 10 )) 秒）"; ready=1; break
  fi
  if ! pgrep -f "vllm serve" >/dev/null; then
    echo "  ✗ 进程已退出（$(( i * 10 )) 秒内）"; break
  fi
  [ $(( i % 6 )) -eq 0 ] && echo "    …$(( i * 10 ))s｜$(tail -1 "$LOG" 2>/dev/null | cut -c1-100)"
done

echo; echo "=== 起服日志尾 25 行 ==="
tail -25 "$LOG" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 关键行 ==="
grep -E "Resolved architecture|model weights took|GPU KV cache|Using .* backend|Available routes|Starting vLLM|torch.compile" "$LOG" 2>/dev/null | tail -10 | sed 's/^/  /'

echo; echo "=== 服务端自证（model id） ==="
curl -s --max-time 10 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | sed 's/^/  /'
echo

echo "=== 实际加载的权重路径（从日志确认是 bf16 目录） ==="
grep -m2 -o "/root/rivermind-data/models/[A-Za-z0-9._-]*" "$LOG" 2>/dev/null | sort -u | sed 's/^/  /'

echo; echo "=== GPU 占用 ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/  /'

echo; echo "=== 就绪后下一步 ==="
[ "$ready" = "1" ] && cat <<'NOTE'
  ① 接线自校验：python -B /root/rivermind-data/_smoke_qwen32b.py
  ② 20 条目检
  ③ H 的枚举测量（154 条水平轴子类）
  ④ G 全量 4587 条非序数
NOTE
echo "=== 结束 ==="