#!/bin/bash
# 小样续跑：**不删产物**，只补缺失的 10 条（depth 加载器 .jpg 组 bug 已修）
set -u
R=/root/rivermind-data
OUT=$R/output/qwen32b_d3_sample.json
OUTS=$R/output/qwen32b_d3_sample_som.json
LOG=$R/d_sample_resume.log

echo "=== 续跑前状态 ==="
"$R/venv/bin/python" - "$OUT" "$R/_d_sample300.json" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = set(json.load(open(sys.argv[2], encoding="utf-8")))
print(f"  产物 {len(d)} 条｜清单 {len(s)}｜缺 {len(s - set(d))} 条"
      f"｜清单外 {len(set(d) - s)} 条")
PYEOF

echo; echo "=== 续跑（PHASE=A / 模态=3 / 范围=小样）==="
cd "$R"
QWEN_ENGINE=api32b \
QWEN_MODALITIES=3 \
QWEN_PHASE=A \
QWEN_ONLY_NONORD=1 \
QWEN_QIDS=$R/_d_sample300.json \
QWEN_DATA=$R/dataset \
QWEN_CORES=$R/_ordinal_cores_v3.json \
QWEN_OUT_DIRECT="$OUT" \
QWEN_OUT_SOM="$OUTS" \
QWEN_CONC=8 \
"$R/venv/bin/python" -u "$R/_qwen_full_pipeline.py" > "$LOG" 2>&1
echo "  管线退出码 $?"
echo "  日志尾: $(tail -1 $LOG)"

echo; echo "=== 续跑后清点 ==="
"$R/venv/bin/python" - "$OUT" "$R/_d_sample300.json" <<'PYEOF'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = set(json.load(open(sys.argv[2], encoding="utf-8")))
errs = {k for k, v in d.items() if v.get("err")}
print(f"  产物 {len(d)} 条｜清单 {len(s)}｜缺 {len(s - set(d))}"
      f"｜清单外 {len(set(d) - s)}｜err {len(errs)}")
nb = sum(1 for v in d.values() if v.get("bbox"))
print(f"  有框 {nb}｜输出[]无框 {len(d) - nb - len(errs)}")
if errs:
    for k in sorted(errs)[:5]:
        print(f"    err {k}: {d[k]['err'][:60]}")
PYEOF
echo "=== 结束 ==="