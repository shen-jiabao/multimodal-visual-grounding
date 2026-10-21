#!/bin/bash
# 取 vLLM 完整报错上下文（定位 nvcc 为何仍找不到）
set -u
R=/root/rivermind-data
L=$R/vllm38.log

echo "=== 报错全段（前后各 25 行）==="
grep -n "Could not find nvcc" "$L" 2>/dev/null | head -2 | cut -d: -f1 | while read -r ln; do
  st=$(( ln > 25 ? ln - 25 : 1 ))
  sed -n "${st},$(( ln + 12 ))p" "$L" | cut -c1-190 | sed 's/^/  /'
  echo "  ---"
done

echo; echo "=== 所有 ERROR 行 ==="
grep -nE "ERROR" "$L" 2>/dev/null | head -12 | cut -c1-190 | sed 's/^/  /'

echo; echo "=== Traceback 尾 ==="
grep -nE "File \"|Error" "$L" 2>/dev/null | tail -18 | cut -c1-190 | sed 's/^/  /'

echo; echo "=== 进程与 GPU ==="
echo "  vllm $(pgrep -cf 'vllm serve' || echo 0)｜serve_run $(pgrep -cf _q38_serve_run.sh || echo 0)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo; echo "=== 手动验证 CUDA_HOME 是否被 torch 认到 ==="
V=$R/venv_q38
PIPCUDA="$V/lib/python3.10/site-packages/nvidia/cu13"
CUDA_HOME="$PIPCUDA" PATH="$PIPCUDA/bin:$PATH" "$V/bin/python" - <<'PY' 2>&1 | sed 's/^/  /'
import os
print("  env CUDA_HOME =", os.environ.get("CUDA_HOME"))
from torch.utils.cpp_extension import CUDA_HOME as CH
print("  torch.utils.cpp_extension.CUDA_HOME =", CH)
import shutil
print("  shutil.which('nvcc') =", shutil.which("nvcc"))
try:
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    print("  _get_cuda_arch_flags ok")
except Exception as e:
    print("  arch flags ✗", e)
PY
echo "=== 结束 ==="