#!/bin/bash
# 侦察：这台机器是否保留了上次 32B 全量跑的成果（模型 / venv / 数据 / 产物）
set -u
echo "=== 主机 ==="; hostname; date
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv 2>/dev/null || echo "  无 nvidia-smi"
echo; echo "=== 磁盘 ==="
df -h 2>/dev/null | grep -Ev 'tmpfs|overlay|udev' | head -12
echo; echo "=== 关键路径是否幸存 ==="
for p in /root/rivermind-data \
         /root/rivermind-data/models \
         /root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ \
         /root/rivermind-data/venv \
         /root/rivermind-data/dataset \
         /root/rivermind-data/output \
         /root/rivermind-data/output/qwen32b_direct.json \
         /root/rivermind-data/_ordinal_cores_v3.json \
         /root/rivermind-data/_qwen_full_pipeline.py ; do
  if [ -e "$p" ]; then
    echo "  [有] $p  ($(du -sh "$p" 2>/dev/null | cut -f1))"
  else
    echo "  [无] $p"
  fi
done
echo; echo "=== 运行中进程 ==="
ps aux | grep -E 'vllm|_qwen_full' | grep -v grep | head -5 || true
echo "  (以上为空表示无在跑任务)"
echo; echo "=== venv 包 ==="
V=/root/rivermind-data/venv/bin/python
if [ -x "$V" ]; then
  $V -c "import sys;print('  python', sys.version.split()[0])"
  for m in vllm transformers torch openai PIL numpy; do
    $V -c "import $m;print('  $m', getattr($m,'__version__','?'))" 2>/dev/null \
      || echo "  $m 缺失"
  done
else
  echo "  无 venv"
fi
echo; echo "=== 数据完整性抽查 ==="
D=/root/rivermind-data/dataset
if [ -d "$D" ]; then
  for m in visible infrared depth; do
    n=$(ls "$D/Images/$m" 2>/dev/null | wc -l)
    echo "  Images/$m : $n 张"
  done
  [ -f "$D/queries/queries.json" ] && echo "  queries.json 有"
else
  echo "  数据集目录不存在"
fi
echo; echo "=== 网络 ==="
timeout 12 curl -sI https://www.modelscope.cn 2>/dev/null | head -1 || echo "  modelscope 不可达"
echo "=== 侦察结束 ==="