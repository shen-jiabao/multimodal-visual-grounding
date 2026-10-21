#!/bin/bash
# 诊断：为何 32B 有框率偏低（197/227 = 86.8%，而 8B 是 99.2%）。
# 看无框条目的 raw 输出形态，判断是「格式不同」还是「模型真的拒答」。
PY=/root/rivermind-data/venv/bin/python
$PY - <<'PYEOF'
import json
d = json.load(open("/root/rivermind-data/output/qwen32b_direct.json", encoding="utf-8"))
nobox = [(k, v) for k, v in d.items() if not v.get("bbox")]
print("总 " + str(len(d)) + " 条 | 无框 " + str(len(nobox)) + " 条 ("
      + format(len(nobox)/max(len(d),1), ".1%") + ")")
print()
print("=== 无框条目的 raw 形态（前 12 条）===")
for k, v in nobox[:12]:
    raw = (v.get("raw") or "").replace("\n", " ")
    print("  " + k + " n=" + str(v.get("n")) + " raw=" + raw[:150])
print()
# 统计无框 raw 里的关键词，判断是哪种失败
import collections
c = collections.Counter()
for k, v in nobox:
    raw = (v.get("raw") or "").lower()
    if not raw:
        c["空输出"] += 1
    elif "none" in raw:
        c["含 none"] += 1
    elif "box" in raw or "bbox" in raw:
        c["提到 box 但未被正则匹配"] += 1
    elif "no " in raw or "cannot" in raw or "not" in raw:
        c["否认/拒答类"] += 1
    else:
        c["其他"] += 1
print("=== 无框原因分类 ===")
for k2, v2 in c.most_common():
    print("  " + k2 + ": " + str(v2))
print()
print("=== 有框条目的 raw 形态（前 3 条，作对照）===")
for k, v in [x for x in d.items() if x[1].get("bbox")][:3]:
    print("  " + k + " raw=" + (v.get("raw") or "").replace("\n", " ")[:150])
PYEOF