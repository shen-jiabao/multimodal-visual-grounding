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
