#!/bin/bash
# 建 /usr/local/cuda 符号链接（flashinfer 默认回退路径）+ 生成起服环境文件
# 去掉上一次那个会永久阻塞的 multiprocessing 测试
set -u
R=/root/rivermind-data
V=$R/venv_q38
PIPCUDA="$V/lib/python3.10/site-packages/nvidia/cu13"

echo "=== ① flashinfer 在显式环境下能否解析 cuda 路径 ==="
CUDA_HOME="$PIPCUDA" PATH="$PIPCUDA/bin:$PATH" "$V/bin/python" -c "
from flashinfer.jit.cpp_ext import get_cuda_path
print('  显式 CUDA_HOME 时 get_cuda_path() =', get_cuda_path())" 2>&1 | tail -3 | sed 's/^/  /'

echo; echo "=== ② JIT 依赖 ==="
for c in ninja c++ g++ gcc which; do printf "  %-8s %s\n" "$c" "$(command -v $c || echo '缺')"; done

echo; echo "=== ③ 建符号链接 /usr/local/cuda → pip cu13 ==="
if [ -L /usr/local/cuda ] || [ -e /usr/local/cuda ]; then
  echo "  已存在：$(ls -ld /usr/local/cuda | cut -c1-110)"
else
  mkdir -p /usr/local
  ln -s "$PIPCUDA" /usr/local/cuda && echo "  ✓ 已建：/usr/local/cuda → $PIPCUDA"
fi
echo "  nvcc 可达性：$(ls -l /usr/local/cuda/bin/nvcc 2>&1 | tail -1 | cut -c1-90)"

echo; echo "=== ④ ★复验：不设 CUDA_HOME，只靠符号链接 ==="
env -u CUDA_HOME -u CUDA_PATH "$V/bin/python" -c "
from flashinfer.jit.cpp_ext import get_cuda_path
print('  仅靠符号链接 get_cuda_path() =', get_cuda_path())" 2>&1 | tail -3 | sed 's/^/  /'

echo; echo "=== ⑤ 生成起服环境文件 ==="
cat > "$R/_q38_env.sh" <<EOS
export CUDA_HOME="$PIPCUDA"
export CUDA_PATH="$PIPCUDA"
export PATH="$PIPCUDA/bin:\$PATH"
export FLASHINFER_NVCC="$PIPCUDA/bin/nvcc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOS
sed 's/^/  /' "$R/_q38_env.sh"

echo; echo "=== ⑥ 当前 vllm 进程 / GPU ==="
echo "  vllm $(pgrep -cf 'vllm serve' || echo 0)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="