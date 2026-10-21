#!/bin/bash
# 侦察（只读）：GPU / 磁盘 / 既有资产 / venv —— 决定下一步能用什么模型
set -u
R=/root/rivermind-data

echo "=== GPU ==="
if ls /dev/nvidia* >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>&1 \
    | sed 's/^/  /'
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | sed 's/^/  驱动 /'
else
  echo "  ★ 无 GPU 设备节点（/dev/nvidia* 为空）"
fi

echo; echo "=== 磁盘 ==="
df -h "$R" 2>/dev/null | sed 's/^/  /'
df -h /data 2>/dev/null | tail -1 | sed 's/^/  /'

echo; echo "=== 既有模型 ==="
du -sh "$R"/models/* 2>/dev/null | sed 's/^/  /'

echo; echo "=== 根目录（前 30 项）==="
ls -1 "$R" 2>/dev/null | head -30 | sed 's/^/  /'

echo; echo "=== venv ==="
if [ -x "$R/venv/bin/python" ]; then
  "$R/venv/bin/python" -c \
    "import vllm,transformers,torch;print('vllm',vllm.__version__,'| tf',transformers.__version__,'| torch',torch.__version__,'| cuda',torch.cuda.is_available())" 2>&1 \
    | tail -3 | sed 's/^/  /'
else
  echo "  ★ 无 venv"
fi

echo; echo "=== 数据 ==="
for d in visible infrared depth; do
  n=$(ls -1 "$R/dataset/Images/$d" 2>/dev/null | wc -l)
  echo "  $d: $n 张"
done
echo "  queries.json: $([ -f "$R/dataset/queries/queries.json" ] && echo 有 || echo 无)"
echo "=== 结束 ==="
