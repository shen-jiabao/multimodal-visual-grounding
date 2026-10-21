#!/bin/bash
# 定位 CCCL / nvcc / CUDA headers 的版本不匹配
set -u
V=/root/rivermind-data/venv_q38
PIP=$V/lib/python3.10/site-packages/nvidia/cu13
FI=$V/lib/python3.10/site-packages/flashinfer

echo "=== ① 报错头文件的判定条件（第 20~50 行）==="
sed -n '20,50p' "$FI/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h" 2>/dev/null | cat -n | sed 's/^/  /'

echo; echo "=== ② nvcc 版本 ==="
"$PIP/bin/nvcc" --version 2>&1 | tail -3 | sed 's/^/  /'

echo; echo "=== ③ CUDA 头文件版本（cuda.h 里的宏）==="
for h in "$PIP/include/cuda.h" /usr/local/cuda/include/cuda.h; do
  echo "  --- $h ---"
  grep -m3 -E "define CUDA_VERSION|define CUDART_VERSION" "$h" 2>/dev/null | sed 's/^/    /'
done

echo; echo "=== ④ CCCL 自带版本宏 ==="
grep -rn -m3 -E "define CCCL_VERSION|define _CCCL_VERSION|CCCL_MAJOR|CCCL_MINOR" \
  "$FI/data/cccl/libcudacxx/include/cuda/std/__cccl/version.h" 2>/dev/null | head -6 | sed 's/^/  /'
ls -1 "$FI/data/cccl" 2>/dev/null | sed 's/^/    /'

echo; echo "=== ⑤ 该头文件里 cuda_toolkit 的判定原文 ==="
sed -n '30,46p' "$FI/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h" 2>/dev/null | sed 's/^/  /'

echo; echo "=== ⑥ flashinfer 是否有预编译 cubin（可关 JIT） ==="
"$V/bin/python" -c "
import flashinfer, os
print('  flashinfer', flashinfer.__version__)
try:
    import flashinfer_cubin
    print('  flashinfer_cubin 存在')
except Exception as e:
    print('  flashinfer_cubin:', type(e).__name__)
" 2>&1 | tail -3 | sed 's/^/  /'
echo "=== DONE ==="