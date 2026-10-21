#!/bin/bash
# 诊断：非确定性实验为何中断 + 服务与 GPU 是否还在
set -u
R=/root/rivermind-data

echo "=== ① 进程 ==="
echo "  nondet: $(pgrep -cf '_q32bf16_nondet.py' || true) 个"
echo "  vllm  : $(pgrep -cf 'vllm serve' || true) 个"
pgrep -af "vllm serve" | head -2 | cut -c1-130 | sed 's/^/    /'
echo "  python 进程总数: $(pgrep -cf python || true)"

echo; echo "=== ② GPU ==="
nvidia-smi 2>&1 | head -12 | sed 's/^/  /'
echo "  exit=$?"
ls /dev/nvidia[0-9]* 2>/dev/null | sed 's/^/  dev /'

echo; echo "=== ③ 完整日志（含结尾，找中断原因） ==="
wc -l "$R/_nondet.log" 2>/dev/null | sed 's/^/  /'
tail -20 "$R/_nondet.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== ④ 服务端存活探测（3 秒超时） ==="
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/v1/models 2>/dev/null || echo ERR)
echo "  /v1/models → $code"

echo; echo "=== ⑤ 内核 OOM / 杀进程痕迹 ==="
dmesg 2>/dev/null | tail -20 | grep -iE "killed process|oom|out of memory" | tail -6 | sed 's/^/  /' || echo "  （无 dmesg 或无线索）"

echo; echo "=== ⑥ 磁盘 ==="
df -h $R | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="