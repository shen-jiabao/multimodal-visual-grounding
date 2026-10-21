#!/bin/bash
# 记下官方 sha256 供下载完成后校验 + 复查 GPU
set -u
R=/root/rivermind-data
echo "model.safetensors 20033961788 abbbc2d103d76bf00b032bb09a9b833ff089890a763c4874a7f28f98c62af351" \
  > "$R/_vlx_sha.txt"
echo "=== 已写入 $R/_vlx_sha.txt ==="
cat "$R/_vlx_sha.txt" | sed 's/^/  /'

echo; echo "=== GPU 复查 ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
             --format=csv,noheader 2>&1 | sed 's/^/  /'
else
  echo "  ○ 无 nvidia-smi"
fi
c=$(ls -1 /dev/nvidia* 2>/dev/null | wc -l)
echo "  /dev/nvidia* 条目数 = $c"
echo "=== 结束 ==="
