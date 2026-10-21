#!/bin/bash
# D 小样：300 条（作用范围内分层抽样）三模态输入 vs 单图基线
# 判据见 `_prereg_D.md` §5；本脚本只负责跑与清点，判定在本地做。
set -u
R=/root/rivermind-data
OUT=$R/output/qwen32b_d3_sample.json
OUTS=$R/output/qwen32b_d3_sample_som.json
LOG=$R/d_sample_run.log

mkdir -p "$R/output"
# 小样必须是**干净的一次性测量**——不续跑任何旧产物（否则改动率会被历史结果污染）
rm -f "$OUT" "$OUTS"

echo "=== 开始：模态=3 阶段=A 仅非序数 作用范围=小样 300 条 ==="
date
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
rc=$?
echo "管线退出码 $rc"
echo; echo "=== 日志（末 8 行）==="
tail -8 "$LOG"
echo; echo "=== 产物清点 ==="
"$R/venv/bin/python" - "$OUT" <<'PYEOF'
import json, sys
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
nb = sum(1 for v in d.values() if v.get("bbox"))
ne = sum(1 for v in d.values() if v.get("err"))
print(f"  条数 {len(d)}｜有框 {nb}｜无框 {len(d)-nb}｜err {ne}")
if ne:
    from collections import Counter
    c = Counter(v["err"][:60] for v in d.values() if v.get("err"))
    for k, n in c.most_common(3):
        print(f"    err × {n}: {k}")
PYEOF
echo "=== 结束 ==="