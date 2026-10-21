#!/bin/bash
# 磁盘与安装条件细查：30 GB 是否够放「vLLM + 32B-AWQ(约 20 GB) + 数据(3.93 GB)」。
echo "=== 全部挂载点 ==="
df -h
echo
echo "=== 块设备 ==="
lsblk 2>/dev/null | head -20
echo
echo "=== 各候选数据盘的可用量 ==="
for d in / /root /data /workspace /mnt /tmp /home; do
  [ -d "$d" ] && echo "  $(df -h "$d" 2>/dev/null | tail -1 | awk '{print $2" 总 / "$4" 可用  ->  '$d'}')"
done
echo
echo "=== pip / conda 可用性 ==="
python3 -m pip --version 2>&1 | head -2
which conda mamba 2>/dev/null || echo "  无 conda/mamba"
echo
echo "=== 网络：能否访问 ModelScope / PyPI（决定下载可行性）==="
for u in https://pypi.org/simple/ https://www.modelscope.cn https://hf-mirror.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 12 "$u" 2>/dev/null)
  echo "  $u -> HTTP $code"
done
echo
echo "=== CUDA / 驱动 ==="
nvidia-smi --query-gpu=compute_cap --format=csv 2>/dev/null
ldconfig -p 2>/dev/null | grep -c libcuda
echo
echo "=== 已装的 apt 级组件（是否有 gcc/git）==="
for c in gcc git curl wget tmux; do command -v $c >/dev/null && echo "  有 $c" || echo "  无 $c"; done