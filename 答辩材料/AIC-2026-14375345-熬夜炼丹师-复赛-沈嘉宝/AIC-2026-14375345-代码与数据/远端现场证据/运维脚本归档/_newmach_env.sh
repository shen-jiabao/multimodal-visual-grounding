#!/bin/bash
# 在新机生成统一环境文件 $R/_q38_env.sh。
#
# 为什么要这个文件：`_rc.py` 走 `bash -s`（**非交互 shell**），~/.bashrc 不会被加载
# ⇒ python3 / nvcc 都不在 PATH。各脚本在开头 source 本文件即可拿到一致环境。
# （旧平台靠系统 python3 在 PATH 里侥幸不用它；新机的 python 在 conda 里，必须显式给。）
set -u

R=/root/rivermind-data
F=$R/_q38_env.sh

cat > "$F" <<'EOF'
# ===== 新机统一环境（由 _newmach_env.sh 生成；各脚本开头 source 本文件）=====
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH=/usr/local/cuda
export HF_HOME=/root/rivermind-data/hf
export HF_HUB_CACHE=/root/rivermind-data/hf/hub
export PIP_CACHE_DIR=/root/rivermind-data/pipcache
export TMPDIR=/root/rivermind-data/tmp
export XDG_CACHE_HOME=/root/rivermind-data/cache
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ===== 结束 =====
EOF

# shellcheck disable=SC1090
. "$F"

echo "✓ 已生成 $F"
echo "  python3 : $(command -v python3)  $(python3 -V 2>&1)"
echo "  nvcc    : $(command -v nvcc)  $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*')"
echo "  pip     : $(command -v pip3 || echo '（无 pip3，用 python3 -m pip）')"
echo "  HF      : $HF_ENDPOINT"
echo "  TMPDIR  : $TMPDIR"
echo
echo "  磁盘："
df -h /root/rivermind-data | tail -1 | sed 's/^/    /'
echo "  解析：$(readlink -f /root/rivermind-data)"
echo
echo "=== DONE ==="
