#!/bin/bash
# 接线自校验：用**新起的 bf16 服务**重跑 AWQ 那次的同一批探针，看接线是否正常
#   · 引擎 api32b，模型名改成 qwen3-vl-32b-bf16（AWQ 那次是 qwen3-vl-32b）
#   · --mode both：既测自身可解析性，也与 AWQ 参考（_probe_32b.json）比对
set -u
R=/root/rivermind-data
cd $R

echo "=== 0) 探针文件是否在 ==="
for f in /root/_probe_qids.json /root/_probe_32b.json; do
  if [ -f "$f" ]; then echo "  ✓ $f（$(wc -c < "$f") 字节）"; else echo "  ✗ 缺 $f"; fi
done
echo "  服务端模型："; curl -s --max-time 8 http://127.0.0.1:8000/v1/models | head -c 200 | sed 's/^/    /'
echo

echo "=== 1) 接线自校验（n=20，与 AWQ 参考对比） ==="
QWEN_ENGINE=api32b \
QWEN_API_MODEL=qwen3-vl-32b-bf16 \
QWEN_DATA=$R/dataset \
QWEN_CORES=$R/_ordinal_cores_r2.json \
  timeout 900 $R/venv/bin/python -u _smoke_qwen32b.py --mode both --n 20 2>&1 \
  | tail -40 | cut -c1-190 | sed 's/^/  /'

echo; echo "=== 2) GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="