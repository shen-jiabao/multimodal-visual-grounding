#!/bin/bash
# Qwen3.8-27B 闸门 G2/G3 冒烟（20 条）—— 预登记 I §4
#   G2 机检：可解析 ≥16/20      G3 人眼：方向正确 ≥12 通过｜8~11 放行登记"接地能力弱"｜<8 停
set -u
R=/root/rivermind-data
# 客户端解释器：优先旧 venv（原单变量口径），新机没有时回退 venv_q38（同含 openai/pillow）
PY=$R/venv/bin/python
[ -x "$PY" ] || PY=$R/venv_q38/bin/python
echo "客户端解释器: $PY"
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
# ★qid 清单：新机在 $R 下（旧机在 /root），探测两处
QIDSFILE=$R/_probe_qids.json
[ -f "$QIDSFILE" ] || QIDSFILE=/root/_probe_qids.json
echo "  qid 清单: $QIDSFILE ($([ -f "$QIDSFILE" ] && echo 存在 || echo ★缺失))"
QWEN_ENGINE=api32b QWEN_API_BASE="$API" QWEN_API_MODEL="$MODEL" \
QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json \
  timeout 1800 $PY -u _q38_gate.py --n 20 --qids "$QIDSFILE" 2>&1 | cut -c1-200 | sed 's/^/  /'
echo "  退出码 ${PIPESTATUS[0]}"

echo; echo "=== 2) 产物 ==="
ls -1 "$R/q38_gate_vis" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="
