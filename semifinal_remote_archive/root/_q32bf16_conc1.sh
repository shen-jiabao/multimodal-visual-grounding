#!/bin/bash
# 接线判据【同条件复现】：把我方管线也设成串行（CONC=1），与归档实现条件一致
#
# 目的：上一轮 16/20 逐位相同、4 条只差坐标末位（≤0.002）。怀疑来源是**批量并发**
#   造成的浮点抖动（我方 CONC>1，归档实现串行）。若串行化后偏差消失 ⇒ 接线本身无错，
#   闸门可在其自身口径下通过；若偏差仍在 ⇒ 说明确实存在接线差异，不得放行。
set -u
R=/root/rivermind-data
cd $R

echo "=== 同条件复现：QWEN_CONC=1 ==="
QWEN_ENGINE=api32b QWEN_API_MODEL=qwen3-vl-32b-bf16 QWEN_DATA=$R/dataset \
QWEN_CORES=$R/_ordinal_cores_r2.json QWEN_CONC=1 \
  timeout 900 $R/venv/bin/python -u _smoke_qwen32b.py --mode both --n 20 2>&1 \
  | tail -30 | cut -c1-190 | sed 's/^/  /'

echo; echo "=== 对照：用归档实现自比（自己 vs 自己，确认服务端本身非确定） ==="
echo "  （若同一实现跑两遍也不逐位相同，则"逐位相同"这个判据在非确定服务端上不成立）"
echo "=== 结束 ==="