#!/bin/bash
# Qwen3.8-27B 闸门 G2/G3 冒烟（20 条）—— 预登记 I §4
#   G2 机检：可解析 ≥16/20      G3 人眼：方向正确 ≥12 通过｜8~11 放行登记"接地能力弱"｜<8 停
set -u
R=/root/rivermind-data
PY=$R/venv/bin/python          # HTTP 客户端沿用旧 venv（与 bf16 那次同一条，保持单变量）
API=http://127.0.0.1:8000/v1
MODEL=qwen38-27b

echo "=== 0) 前置 ==="
curl -s --max-time 8 "$API/models" | head -c 250 | sed 's/^/  服务 /'; echo
for f in "$R/_qwen_full_pipeline.py" "$R/_probe_qids.json" \
         "$R/_ordinal_cores_r2.json" "$R/dataset/queries/queries.json"; do
  [ -f "$f" ] && echo "  ✓ $f" || echo "  ★缺 $f"
done
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'

echo; echo "=== 1) 跑 20 条（direct；temperature=0） ==="
cd "$R"
QWEN_ENGINE=api32b QWEN_API_BASE="$API" QWEN_API_MODEL="$MODEL" \
QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
  timeout 1800 $PY -u _q38_gate.py --n 20 2>&1 | cut -c1-200 | sed 's/^/  /'
echo "  退出码 ${PIPESTATUS[0]}"

echo; echo "=== 2) 产物 ==="
ls -1 "$R/q38_gate_vis" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="
