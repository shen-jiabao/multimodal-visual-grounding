#!/bin/bash
# 取完整 Ninja 失败的链接报错
set -u
R=/root/rivermind-data
L=$R/vllm38.log

echo "=== Ninja 失败全段（从 'Ninja build failed' 起 45 行）==="
grep -n "Ninja build failed" "$L" 2>/dev/null | head -1 | cut -d: -f1 | while read -r ln; do
  sed -n "${ln},$(( ln + 45 ))p" "$L" | cut -c1-200 | sed 's/^/  /'
done

echo; echo "=== 报错关键字 ==="
grep -nE "undefined reference|cannot find|No such file|ld:|collect2|libcudart|libcuda|libnvrtc|-lcuda|/usr/local/cuda" "$L" 2>/dev/null | tail -14 | cut -c1-200 | sed 's/^/  /'

echo; echo "=== CUDA 目录里 lib / lib64 的实况 ==="
V=/root/rivermind-data/venv_q38
CU=$V/lib/python3.10/site-packages/nvidia/cu13
ls -ld "$CU"/lib "$CU"/lib64 2>&1 | sed 's/^/  /'
echo "  lib 下关键库："
ls -1 "$CU/lib" 2>/dev/null | grep -E "cudart|cuda\.so|nvrtc" | head -6 | sed 's/^/    /'

echo; echo "=== 当前后端重试状态 ==="
echo "  vllm $(pgrep -cf 'vllm serve' || echo 0)｜serve_run $(pgrep -cf _q38_serve_run.sh || echo 0)"
tail -3 "$R/_q38_serve.log" | cut -c1-150 | sed 's/^/  /'
echo "=== DONE ==="