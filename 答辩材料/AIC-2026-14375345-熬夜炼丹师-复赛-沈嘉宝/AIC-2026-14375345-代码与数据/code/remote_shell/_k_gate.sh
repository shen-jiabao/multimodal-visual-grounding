#!/bin/bash
# K 件闸门 G2/G3 冒烟（20 条）
#   与 I 件**同一批 20 条**（复用 _q38_gate20.json 的键序），保证两件可比。
set -u
R=/root/rivermind-data
PY=$R/venv/bin/python
export QWEN_API_MODEL=qwen25vl-72b-awq
export QWEN_API_BASE=http://127.0.0.1:8000/v1
export QWEN_DATA=$R/dataset
export QWEN_CORES=$R/_ordinal_cores_r2.json
# ★ 输出适配：Qwen2.5-VL 输出的是它自己 smart-resize 后的像素坐标（不是 0-1000）
export QWEN_BBOX_SPACE=smart_resize_px
QIDS=/root/_probe_qids.json

echo "===== 0) 前置 ====="
[ -f "$R/_q38_gate.py" ] && echo "  ✓ $R/_q38_gate.py" || { echo "  ★FAIL 缺 _q38_gate.py"; exit 9; }
[ -f "$R/_qwen_full_pipeline.py" ] && echo "  ✓ _qwen_full_pipeline.py" || { echo "  ★FAIL 缺管线"; exit 9; }
if [ ! -f "$QIDS" ]; then
  echo "  $QIDS 不存在 ⇒ 用 I 件闸门结果的键序重建（同一批 20 条）"
  $PY - <<'PY'
import json, os
p = "/root/rivermind-data/_q38_gate20.json"
if not os.path.exists(p):
    raise SystemExit("★ 也没有 _q38_gate20.json，无法保证可比")
ks = list(json.load(open(p, encoding="utf-8")).keys())
json.dump(ks, open("/root/_probe_qids.json", "w"), ensure_ascii=False)
print(f"  已重建 {len(ks)} 条")
PY
  [ -f "$QIDS" ] || { echo "  ★FAIL 无法确定冒烟条目"; exit 9; }
fi
echo "  条目文件 $QIDS（$($PY -c "import json;print(len(json.load(open('$QIDS'))))") 条）"

echo "===== 1) 服务自证 ====="
curl -s --max-time 8 "$QWEN_API_BASE/models" | tr ',' '\n' | grep -E '"id"|"root"' | sed 's/^/  /'

echo "===== 2) 跑 20 条冒烟 ====="
$PY "$R/_q38_gate.py" --qids "$QIDS" --data "$R/dataset" \
    --out "$R/_k_gate20.json" --vis-dir "$R/k_gate_vis" --n 20
echo "=== GATE DONE ==="
