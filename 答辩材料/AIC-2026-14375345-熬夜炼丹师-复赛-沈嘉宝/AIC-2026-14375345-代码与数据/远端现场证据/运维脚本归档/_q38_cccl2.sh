#!/bin/bash
# ① 找 flashinfer 传 nvcc 标志的环境变量名  ② 试 FLASHINFER_DISABLE_JIT=1 起服
set -u
R=/root/rivermind-data
V=$R/venv_q38
FI=$V/lib/python3.10/site-packages/flashinfer

echo "=== ① flashinfer 可用的环境变量（cpp_ext.py 里的 getenv 名）==="
grep -nE "os\.environ\.get\(|getenv\(" "$FI/jit/cpp_ext.py" 2>/dev/null | head -20 | sed 's/^/  /'
echo "  --- 含 FLAGS 的环境变量名 ---"
grep -rhoE "FLASHINFER_[A-Z_]*(FLAGS|CFLAGS|NVCC|EXTRA)[A-Z_]*" "$FI/" 2>/dev/null | sort -u | head -12 | sed 's/^/  /'
echo "  --- _get_env_flags / _get_glibcxx 附近的定义（第 1~30 行）---"
sed -n '1,30p' "$FI/jit/cpp_ext.py" 2>/dev/null | grep -nE "def |environ|FLASHINFER" | sed 's/^/  /'

echo; echo "=== ② 提交一次 JIT 编译测试（带 CCCL 开关，验证该开关能否绕过错配）==="
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH \
FLASHINFER_NVCC=/usr/local/cuda/bin/nvcc \
  timeout 120 make -n >/dev/null 2>&1; echo "  （占位）"

echo; echo "=== ③ 直接用 nvcc 编译一个最小 CUDA 文件，复现/验证该错 ==="
mkdir -p /tmp/_cccl_t && cd /tmp/_cccl_t
cat > t.cu <<'EOF'
#include <cuda_runtime_api.h>
#include <cstdio>
int main(){ printf("CUDART_VERSION=%d\n", CUDART_VERSION); return 0; }
EOF
cat > t2.cu <<'EOF'
#define CCCL_DISABLE_CTK_COMPATIBILITY_CHECK 1
#include <cuda_runtime_api.h>
#include <cstdio>
int main(){ printf("with-CCCL-off CUDART_VERSION=%d\n", CUDART_VERSION); return 0; }
EOF
echo "  --- 不带头文件（看 CUDART_VERSION 真值）---"
CUDA_HOME=/usr/local/cuda /usr/local/cuda/bin/nvcc -I/usr/local/cuda/include t.cu -o t 2>&1 | head -4 | sed 's/^/    /'
[ -x ./t ] && ./t | sed 's/^/    /'
echo "  --- CCCL 检查关不上（宏在 CCCL 里，需在编译单元最早定义）---"
CUDA_HOME=/usr/local/cuda /usr/local/cuda/bin/nvcc -I/usr/local/cuda/include \
  -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK t.cu -o t2 2>&1 | head -4 | sed 's/^/    /'
[ -x ./t2 ] && ./t2 | sed 's/^/    /'

echo; echo "=== ④ 检查 pip 里 nvcc 与 runtime 的版本（确认错配来源）==="
"$V/bin/python" -m pip list 2>/dev/null | grep -iE "nvidia-cuda-nvcc|nvidia-cuda-runtime|nvidia-cuda-crt|nvidia-cuda-nvvm" | sed 's/^/  /'
echo "=== DONE ==="