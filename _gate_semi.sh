#!/bin/bash
# 复赛全量启动闸门：数据不齐则拒绝启动（exit 1）。可反复运行，无副作用。
PY=/root/miniconda3/envs/myconda/bin/python
DATA=/srv/dataset
OUT=/srv/output
MODEL=/srv/models/Qwen3-VL-8B-Instruct
fail=0

echo "===== 闸门 1) 模型权重 ====="
n_sf=$(ls $MODEL/*.safetensors 2>/dev/null | wc -l)
n_inc=$(ls $MODEL/*.incomplete 2>/dev/null | wc -l)
echo "safetensors=$n_sf  incomplete=$n_inc"
if [ "$n_sf" -lt 4 ] || [ "$n_inc" -ne 0 ]; then echo "FAIL: 模型权重不完整"; fail=1; fi

echo "===== 闸门 2) queries + 核心词 ====="
$PY - <<'PYEOF'
import json
from pathlib import Path
DATA = Path("/srv/dataset"); OUT = Path("/srv/output")
q = json.loads((DATA/"queries"/"queries.json").read_text(encoding="utf-8"))
print("queries", len(q))
c = json.loads((OUT/"_ordinal_cores_v3.json").read_text(encoding="utf-8"))
print("cores", len(c), "| subset_of_queries:", set(c) <= set(q))
PYEOF

echo "===== 闸门 3) 引用图 存在性 + 可读性 ====="
$PY - <<'PYEOF'
import json, sys
from pathlib import Path
from PIL import Image
DATA = Path("/srv/dataset")
q = json.loads((DATA/"queries"/"queries.json").read_text(encoding="utf-8"))
names = sorted({Path(v["visible"]).name for v in q.values()})
vd = DATA/"Images"/"visible"
missing = [n for n in names if not (vd/n).exists()]
corrupt = []
for n in names:
    if n in missing: continue
    try:
        with Image.open(vd/n) as im: im.verify()
    except Exception as e:
        corrupt.append((n, type(e).__name__))
print(f"引用图 {len(names)} | 缺失 {len(missing)} | 损坏 {len(corrupt)}")
if missing:  print("缺失样例", missing[:5])
if corrupt:  print("损坏样例", corrupt[:5])
sys.exit(1 if (missing or corrupt) else 0)
PYEOF
[ $? -ne 0 ] && fail=1

echo "===== 闸门 4) 管线常量（确认写正式目录）====="
grep -nE '^(MODEL_PATH|DATA|OUT_DIRECT|OUT_SOM|CORES|SAVE_EVERY)' /root/_qwen_full_pipeline.py

echo "===== 结论 ====="
if [ $fail -eq 0 ]; then echo "PASS: 可以启动全量"; else echo "BLOCKED: 拒绝启动"; fi
exit $fail
