#!/bin/bash
# 只读：打印 Locate L2(highres slow) 对序数三连的框 + GD 候选池条目
python3 - <<'PY'
import json,os
p="/root/output/predictions_v2_highres_slow.json"
keys=["000023_002","000023_003","000023_004"]
if not os.path.exists(p):
    print("[-] missing", p); raise SystemExit
d=json.load(open(p))
print("L2 file nodes:", len(d))
print("sample keys present:", [k for k in keys if k in d])
for k in keys:
    v=d.get(k)
    if isinstance(v,dict):
        bb=v.get("bbox") or v.get("box") or v
    else:
        bb=v
    print(f"{k} -> {json.dumps(bb, ensure_ascii=False)[:160]}")
PY