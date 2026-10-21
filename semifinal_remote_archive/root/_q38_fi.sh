#!/bin/bash
# 读 flashinfer 的 get_cuda_path 实现（决定该设哪个环境变量或做符号链接）
set -u
V=/root/rivermind-data/venv_q38
F=$V/lib/python3.10/site-packages/flashinfer/jit/cpp_ext.py

echo "=== get_cuda_path 及其上下文（第 30~80 行）==="
sed -n '30,80p' "$F" 2>/dev/null | cat -n | sed 's/^/  /'

echo; echo "=== 该文件里所有 os.environ / getenv 引用 ==="
grep -nE "environ|getenv|which|usr/local" "$F" 2>/dev/null | head -20 | sed 's/^/  /'

echo; echo "=== 是否有其他 CUDA 探测实现（env.py / utils） ==="
grep -rnE "def get_cuda_path|CUDA_HOME|CUDA_PATH" "$V/lib/python3.10/site-packages/flashinfer/" 2>/dev/null | head -14 | sed 's/^/  /'

echo; echo "=== 关键：flashinfer 版本与 JIT 开关 ==="
$V/bin/python -c "import flashinfer;print('  flashinfer', flashinfer.__version__)" 2>&1 | tail -2 | sed 's/^/  /'
echo "  可关闭 JIT 的环境变量（源码里找）："
grep -rnE "FLASHINFER_.*JIT|disable.*jit|JIT.*disable" "$V/lib/python3.10/site-packages/flashinfer/" 2>/dev/null | head -8 | sed 's/^/  /'
echo "=== 结束 ==="