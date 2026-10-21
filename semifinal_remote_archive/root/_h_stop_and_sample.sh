#!/bin/bash
# 停掉「无效枚举」（它跑的是序数条目，H 用不上）+ 打印 154 条轴类查询样本以设计核心词抽取
set -u
R=/root/rivermind-data

echo "=== 1) 停掉无效枚举 ==="
pid=$(pgrep -f _qwen_full_pipeline.py | head -1)
if [ -n "$pid" ]; then kill "$pid" 2>/dev/null && echo "  已停 $pid"; sleep 4; fi
pgrep -f _qwen_full_pipeline.py >/dev/null && { pkill -9 -f _qwen_full_pipeline.py; echo "  已强杀"; } || echo "  已停止"
echo "  vllm 保留：$(pgrep -cf 'vllm serve' || true) 个"

echo; echo "=== 2) 154 条轴类查询（前 24 条，用于设计核心词抽取） ==="
$R/venv/bin/python - "$R/_axis154.json" "$R/dataset/queries/queries.json" <<'PY'
import json, re, sys, collections
ids = json.load(open(sys.argv[1], encoding="utf-8"))
q = json.load(open(sys.argv[2], encoding="utf-8"))
AX = ("leftmost", "rightmost", "topmost", "rearmost")
cnt = collections.Counter()
for k in ids:
    t = q[k]["query"]
    for w in AX:
        if re.search(r"\b"+w+r"\b", t, re.I):
            cnt[w] += 1
            break
print(f"  共 {len(ids)} 条｜分轴 {dict(cnt)}")
for k in ids[:24]:
    print(f"    {k}  {q[k]['query']}")
print("\n  --- 极短与极长的各 4 条（看边界形态）---")
srt = sorted(ids, key=lambda k: len(q[k]["query"]))
for k in srt[:4] + srt[-4:]:
    print(f"    [{len(q[k]['query']):3d}] {k}  {q[k]['query']}")
PY

echo; echo "=== 3) 现有 ordinal 核心词表长什么样（作抽取模板的参照） ==="
$R/venv/bin/python - "$R/_ordinal_cores_r2.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    items = list(d.items())[:6] if isinstance(d, dict) else [(str(i), v) for i, v in enumerate(d[:6])]
    print(f"  类型 {type(d).__name__}｜条数 {len(d)}")
    for k, v in items:
        print(f"    {k} → {v!r}")
except Exception as e:
    print(f"  ★读取失败 {e}")
PY
echo "=== 结束 ==="