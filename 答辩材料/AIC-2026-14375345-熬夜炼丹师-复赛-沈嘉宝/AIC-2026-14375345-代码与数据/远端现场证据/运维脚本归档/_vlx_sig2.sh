#!/bin/bash
# 读 vlx_seek_worker 的编号映射与 predict 主干，确认返回坐标无歧义
set -u
W=/root/rivermind-data/vlx-seek/vlx_seek_worker.py

echo "=== A. 行 100-165（_remap / _build_result_bbox_list / _build_prompt） ==="
sed -n '100,165p' "$W" | cat -n | sed 's/^/  /'

echo; echo "=== B. 行 200-265（predict 主干） ==="
sed -n '200,265p' "$W" | cat -n | sed 's/^/  /'

echo; echo "=== C. run_task 与 grounding_single 模板 ==="
sed -n '285,335p' "$W" | cat -n | sed 's/^/  /'
echo "=== 结束 ==="
