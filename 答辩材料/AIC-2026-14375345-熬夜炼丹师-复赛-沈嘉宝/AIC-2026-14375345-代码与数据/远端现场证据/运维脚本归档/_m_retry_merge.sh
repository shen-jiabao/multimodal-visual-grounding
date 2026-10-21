#!/bin/bash
# 把单条回补结果并进 m_direct.json：**只填原来无框的**，绝不覆盖已有框
set -u
R=/root/rivermind-data
$R/venv_q38/bin/python - "$R" <<'PY'
import json
import os
import sys

R = sys.argv[1]
op = os.path.join(R, "output")
src = os.path.join(op, "m_direct.json")
add = os.path.join(op, "m_retry.json")
dst = os.path.join(op, "m_direct_filled.json")

d = json.load(open(src, encoding="utf-8"))
n_before = sum(1 for v in d.values() if not v.get("bbox"))
a = json.load(open(add, encoding="utf-8")) if os.path.exists(add) else {}
filled = 0
for k, v in a.items():
    if v.get("bbox") and k in d and not d[k].get("bbox"):
        d[k] = v
        filled += 1
json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False)

ok = sum(1 for v in d.values() if v.get("bbox"))
print(f"  回补前无框 {n_before} 条｜回补输入 {len(a)} 条")
print(f"  回补成功 {filled} 条｜仍无框 {n_before - filled} 条")
print(f"  合并后有框 {ok}/{len(d)} = {ok/len(d)*100:.2f}%"
      f"（G4 要求 ≥95% ⇒ {'通过' if ok/len(d) >= 0.95 else '★仍不通过'}）")
print(f"  产物 {dst}")

# 逐位自检：非无框条目必须与原件逐位相同
orig = json.load(open(src, encoding="utf-8"))
moved = [k for k in orig if orig[k].get("bbox") and d[k] != orig[k]]
print(f"  自检：原本有框的条目被动过 {len(moved)} 条（必须为 0）")
PY
echo "=== DONE ==="
