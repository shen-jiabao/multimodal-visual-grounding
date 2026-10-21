#!/bin/bash
# 补 FLASHINFER_EXTRA_CUDAFLAGS（关 CCCL 版本一致性检查）并重启起服
set -u
R=/root/rivermind-data
V=$R/venv_q38
PIPCUDA="$V/lib/python3.10/site-packages/nvidia/cu13"

echo "=== ① 更新环境文件 ==="
cat > "$R/_q38_env.sh" <<EOS
# Qwen3.8-27B 起服环境（由 _q38_fi4.sh 生成）
# 版本错配（实测）：nvcc 13.2.86 vs CUDA runtime 头文件 13.0.96（CUDART_VERSION=13000）
#   ⇒ CCCL 在 flashinfer JIT 时拒绝编译。
#   flashinfer 提供 FLASHINFER_EXTRA_CUDAFLAGS；CCCL 提供关闭开关。
export CUDA_HOME="$PIPCUDA"
export CUDA_PATH="$PIPCUDA"
export PATH="$PIPCUDA/bin:\$PATH"
export FLASHINFER_NVCC="$PIPCUDA/bin/nvcc"
export FLASHINFER_EXTRA_CUDAFLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOS
cat "$R/_q38_env.sh" | sed 's/^/  /'

echo; echo "=== ② 验证该宏能真正关掉检查（用 flashinfer 自己的 include 路径编译）==="
T=/tmp/_cccl_v; mkdir -p $T; cd $T
cat > t.cu <<'EOF'
#include <cuda_runtime_api.h>
#include <cstdio>
int main(){ printf("CUDART_VERSION=%d\n", CUDART_VERSION); return 0; }
EOF
FI=$V/lib/python3.10/site-packages/flashinfer
INC="-I$PIPCUDA/include -I$FI/data/cccl/libcudacxx/include -I$FI/data/cccl/cub -I$FI/data/cccl/thrust"
echo "  --- 不关检查（应报 incompat）---"
$PIPCUDA/bin/nvcc $INC t.cu -o t0 2>&1 | grep -m2 -E "error|incompatible" | sed 's/^/    /' || echo "    （未复现，可能未纳入 CCCL 头）"
echo "  --- 关检查（应编过）---"
$PIPCUDA/bin/nvcc $INC -DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK t.cu -o t1 2>&1 | grep -m2 -E "error|incompatible" | sed 's/^/    /' || true
[ -x ./t1 ] && echo "    ✓ 编译通过，运行：$(./t1)" || echo "    ✗ 仍未通过"

echo; echo "=== ③ 重启起服（清场 → 后台分离）==="
pkill -f "vllm serve" 2>/dev/null && { echo "  已停旧 vllm"; sleep 8; } || echo "  （无旧 vllm）"
: > "$R/_q38_serve.log"
setsid nohup bash "$R/_q38_serve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 3
echo "  已启动起服；进程 $(pgrep -cf _q38_serve_run.sh || echo 0)"
echo "=== DONE ==="