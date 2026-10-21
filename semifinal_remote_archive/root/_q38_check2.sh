#!/bin/bash
# 补核：① 我的 import 名可能写错（fla 而非 flash_linear_attention）
#       ② GPU 是否还在  ③ vLLM 自带 GDN 是否已够用（看源码依赖）
set -u
R=/root/rivermind-data
V=$R/venv_q38

echo "=== ① flash-linear-attention 的真实导入名 ==="
$V/bin/python -m pip show flash-linear-attention 2>/dev/null | head -3 | sed 's/^/  /'
for m in fla flash_linear_attention; do
  $V/bin/python -c "import $m; print('  ✓ import $m ok，版本', getattr($m,'__version__','?'))" 2>&1 | tail -1 | sed 's/^/  /'
done
echo "  已安装包名筛查："
$V/bin/python -m pip list 2>/dev/null | grep -iE "fla|causal|flash" | sed 's/^/    /'

echo; echo "=== ② GPU 是否还在 ==="
ls /dev/nvidia[0-9]* 2>/dev/null | sed 's/^/  dev /' || echo "  ✗ 无 /dev/nvidia* 设备节点"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | head -4 | sed 's/^/  /'
echo "  nvidia-smi 退出码：$?"
echo "  驱动版本：$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo '取不到')"

echo; echo "=== ③ vLLM 自带 GDN 后端是否自足（看它 import 什么） ==="
GU=$V/lib/python3.10/site-packages
echo "  --- gdn_attn.py 的 import ---"
grep -E "^import |^from " "$GU/vllm/v1/attention/backends/gdn_attn.py" 2>/dev/null | head -12 | sed 's/^/    /'
echo "  --- qwen3_5.py 的 import（是否依赖 fla / causal_conv1d）---"
grep -nE "fla|causal_conv1d|gdn|GatedDelta|humming" "$GU/vllm/model_executor/models/qwen3_5.py" 2>/dev/null | head -10 | sed 's/^/    /'
echo "  --- humming_kernels 是否已装（vLLM 自带线性注意力内核）---"
$V/bin/python -c "import humming_kernels as h; print('  ✓ humming_kernels', getattr(h,'__version__','ok'))" 2>&1 | tail -1 | sed 's/^/  /'
$V/bin/python -m pip list 2>/dev/null | grep -iE "humming|flashinfer" | sed 's/^/    /'
echo "=== 结束 ==="