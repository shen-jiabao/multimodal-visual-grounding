#!/bin/bash
# 定位 venv_q38 里的 pip 版 nvcc，供 vLLM 的 GDN 内核 JIT 使用
set -u
R=/root/rivermind-data
V=$R/venv_q38

echo "=== ① site-packages 里的 nvcc ==="
find "$V/lib/python3.10/site-packages/nvidia" -name "nvcc" -type f 2>/dev/null | head -5 | sed 's/^/  /'
find "$V" -name "nvcc" -type f 2>/dev/null | head -5 | sed 's/^/  /'
echo "  nvidia 子包："
ls -1 "$V/lib/python3.10/site-packages/nvidia" 2>/dev/null | sed 's/^/    /'

echo; echo "=== ② 系统 nvcc / CUDA_HOME ==="
echo "  which nvcc: $(command -v nvcc || echo '无')"
echo "  /usr/local/cuda: $([ -d /usr/local/cuda ] && echo 存在 || echo '不存在')"
ls -d /usr/local/cuda* 2>/dev/null | sed 's/^/  /' || echo "  （无 /usr/local/cuda*）"

echo; echo "=== ③ cuda_nvcc 包结构 ==="
NVCCDIR="$V/lib/python3.10/site-packages/nvidia/cuda_nvcc"
if [ -d "$NVCCDIR" ]; then
  echo "  $NVCCDIR"
  find "$NVCCDIR" -maxdepth 3 -type d 2>/dev/null | head -10 | sed 's/^/    /'
  find "$NVCCDIR" -maxdepth 4 -name "nvcc*" 2>/dev/null | head -6 | sed 's/^/    /'
else
  echo "  ★无 cuda_nvcc 目录"
fi

echo; echo "=== ④ 当前起服进程 ==="
echo "  vllm $(pgrep -cf 'vllm serve' || echo 0)"
tail -4 "$R/vllm38.log" 2>/dev/null | cut -c1-150 | sed 's/^/  /'
echo "=== 结束 ==="