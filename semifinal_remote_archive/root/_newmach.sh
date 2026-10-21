#!/bin/bash
# 新机（AutoDL / RTX PRO 6000 Blackwell 96G / CUDA 13.0 镜像）一次性自检。
# 目的：在装任何东西之前，先确认"镜像与磁盘能不能撑住这套栈"。
set -u

echo "=== 1) 系统 ==="
uname -srm
cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION_ID)=' | sed 's/^/  /'

echo
echo "=== 2) 磁盘与挂载点（★系统盘 30G 与数据盘必须分清）==="
df -h / 2>/dev/null | sed 's/^/  /'
if [ -d /root/autodl-tmp ]; then
  echo "  ✓ 数据盘候选 /root/autodl-tmp 存在"
  df -h /root/autodl-tmp 2>/dev/null | sed 's/^/  /'
else
  echo "  ★ 未找到 /root/autodl-tmp，请确认数据盘挂载点"
  ls -d /root/*tmp* /root/*data* /data 2>/dev/null | sed 's/^/    /'
fi

echo
echo "=== 3) GPU / 驱动 ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv 2>&1 | sed 's/^/  /'

echo
echo "=== 4) CUDA toolkit / nvcc（决定 flashinfer JIT 能不能跑）==="
ls -ld /usr/local/cuda* 2>&1 | sed 's/^/  /'
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -2 | sed 's/^/  /'
else
  echo "  ★ 无 nvcc ⇒ 镜像可能是 runtime 版，flashinfer JIT 会失败"
fi

echo
echo "=== 5) Python / conda ==="
command -v python3 >/dev/null 2>&1 && python3 -V 2>&1
ls -d /root/miniconda3 /opt/conda 2>/dev/null | sed 's/^/  /'
python3 -m pip --version 2>&1 | head -1 | sed 's/^/  /'

echo
echo "=== 6) 路径就位情况 ==="
if [ -e /root/rivermind-data ] || [ -L /root/rivermind-data ]; then
  ls -ld /root/rivermind-data | sed 's/^/  已存在: /'
else
  echo "  尚未创建 /root/rivermind-data（下一步建符号链接）"
fi
echo "  HOME=$HOME"

echo
echo "=== 7) 网络（能否直连 pip / 模型站）==="
curl -s -o /dev/null -w "  pypi  %{http_code}  %{time_total}s\n" \
  --max-time 15 https://pypi.org/simple/ 2>&1
curl -s -o /dev/null -w "  hf    %{http_code}  %{time_total}s\n" \
  --max-time 15 https://huggingface.co 2>&1

echo
echo "=== DONE ==="
