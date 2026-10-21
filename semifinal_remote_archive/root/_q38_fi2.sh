#!/bin/bash
# 诊断：worker 环境是否丢失 CUDA_HOME + 建立 /usr/local/cuda 符号链接 + 复启
set -u
R=/root/rivermind-data
V=$R/venv_q38
PIPCUDA="$V/lib/python3.10/site-packages/nvidia/cu13"

echo "=== ① 环境继承测试（父进程设 CUDA_HOME，看 spawn 子进程是否可见）==="
CUDA_HOME="$PIPCUDA" PATH="$PIPCUDA/bin:$PATH" "$V/bin/python" - <<'PY'
import os, multiprocessing as mp, sys
print("  父 os.environ['CUDA_HOME'] =", os.environ.get("CUDA_HOME"))
def child(q):
    q.put((os.environ.get("CUDA_HOME"), os.environ.get("PATH", "")[:60]))
if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,)); p.start(); p.join()
    print("  子 CUDA_HOME =", q.get()[0])
    print("  子 PATH 前缀 =", q.get()[1] if not q.empty() else "(已取)")
PY

echo; echo "=== ② flashinfer 在这个环境下能否解析 cuda 路径 ==="
CUDA_HOME="$PIPCUDA" PATH="$PIPCUDA/bin:$PATH" "$V/bin/python" -c "
from flashinfer.jit.cpp_ext import get_cuda_path
print('  get_cuda_path() =', get_cuda_path())" 2>&1 | tail -3 | sed 's/^/  /'

echo; echo "=== ③ JIT 依赖：ninja / c++ / which ==="
for c in ninja c++ g++ gcc which; do printf "  %-8s %s\n" "$c" "$(command -v $c || echo '✗ 缺')"; done

echo; echo "=== ④ 建立 /usr/local/cuda 符号链接（flashinfer 的默认回退路径）==="
if [ -e /usr/local/cuda ]; then
  echo "  已存在：$(ls -ld /usr/local/cuda | cut -c1-90)"
else
  mkdir -p /usr/local
  ln -s "$PIPCUDA" /usr/local/cuda && echo "  ✓ 已建符号链接 /usr/local/cuda → $PIPCUDA"
  ls -ld /usr/local/cuda | sed 's/^/  /'
fi
echo "  验证：ls /usr/local/cuda/bin/nvcc → $(ls /usr/local/cuda/bin/nvcc 2>&1 | tail -1)"

echo; echo "=== ⑤ 复验（不设 CUDA_HOME，只靠符号链接）==="
env -u CUDA_HOME "$V/bin/python" -c "
from flashinfer.jit.cpp_ext import get_cuda_path
print('  get_cuda_path() =', get_cuda_path())" 2>&1 | tail -3 | sed 's/^/  /'

echo; echo "=== ⑥ 启动参数补充（写入环境文件，供起服脚本读取）==="
cat > "$R/_q38_env.sh" <<EOS
# Qwen3.8-27B 起服所需环境（由 _q38_fi2.sh 生成）
export CUDA_HOME="$PIPCUDA"
export CUDA_PATH="$PIPCUDA"
export PATH="$PIPCUDA/bin:\$PATH"
export FLASHINFER_NVCC="$PIPCUDA/bin/nvcc"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOS
cat "$R/_q38_env.sh" | sed 's/^/  /'
echo "=== 结束 ==="