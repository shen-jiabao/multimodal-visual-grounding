#!/bin/bash
# 版本对齐可行性 + flashinfer 编译标志环境变量名
set -u
V=/root/rivermind-data/venv_q38
FI=$V/lib/python3.10/site-packages/flashinfer

echo "=== ① 已装 nvidia-cuda-* 版本（找错配源）==="
"$V/bin/python" -m pip list 2>/dev/null | grep -iE "^nvidia" | sed 's/^/  /'

echo; echo "=== ② 可用的匹配版本（让头文件追平 nvcc 13.2）==="
for p in nvidia-cuda-runtime-cu13 nvidia-cuda-nvcc-cu13; do
  echo "  --- $p ---"
  "$V/bin/python" -m pip index versions "$p" 2>&1 | head -3 | sed 's/^/    /'
done

echo; echo "=== ③ flashinfer 支持的环境变量（找能加编译标志的）==="
grep -rhoE '"FLASHINFER_[A-Z0-9_]+"' "$FI/jit/" "$FI/__init__.py" 2>/dev/null | sort -u | head -20 | sed 's/^/  /'

echo; echo "=== ④ _get_env_flags 相关实现（cpp_ext.py 前 30 行）==="
sed -n '1,30p' "$FI/jit/cpp_ext.py" 2>/dev/null | sed 's/^/  /'

echo; echo "=== ⑤ nvcc 与 runtime 头文件的确切版本 ==="
"$V/bin/python" - <<'PY'
import importlib.metadata as md
for p in ("nvidia-cuda-nvcc-cu13", "nvidia-cuda-runtime-cu13", "nvidia-cuda-nvcc", "nvidia-cuda-runtime"):
    try: print(f"  {p:32s} {md.version(p)}")
    except Exception: pass
PY
echo "  cuda.h CUDA_VERSION:"
grep -m1 "define CUDA_VERSION" "$V/lib/python3.10/site-packages/nvidia/cu13/include/cuda.h" 2>/dev/null | sed 's/^/    /'
echo "  cuda_runtime_api.h CUDART_VERSION:"
grep -rm1 "define CUDART_VERSION" "$V/lib/python3.10/site-packages/nvidia/cu13/include/cuda_runtime_api.h" 2>/dev/null | sed 's/^/    /'
echo "=== DONE ==="