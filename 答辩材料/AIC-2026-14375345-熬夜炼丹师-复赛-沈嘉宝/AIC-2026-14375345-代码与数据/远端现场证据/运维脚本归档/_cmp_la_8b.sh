#!/bin/bash
# 对比 LA 多框条目 vs 8B 实际输出（读正在跑的 8B 检查点），判断"取 boxes[0]"是否合理。
PY=/root/miniconda3/envs/myconda/bin/python
KEYS="000512_002 000512_004 000085_001 000085_002 000246_001 000246_012 000001_006"
$PY - <<'PY'
import json, os

def iou(p, q):
    if not p or not q: return 0.0
    x1, y1 = max(p[0], q[0]), max(p[1], q[1])
    x2, y2 = min(p[2], q[2]), min(p[3], q[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    i = (x2-x1)*(y2-y1)
    u = (p[2]-p[0])*(p[3]-p[1]) + (q[2]-q[0])*(q[3]-q[1]) - i
    return i/u if u > 0 else 0.0

d = json.load(open("/srv/output/qwen3vl_all.json", encoding="utf-8"))
s = {}
if os.path.exists("/srv/output/qwen3vl_som_all.json"):
    s = json.load(open("/srv/output/qwen3vl_som_all.json", encoding="utf-8"))
try:
    la = json.load(open("/srv/probe_locate/out/probe_result.json", encoding="utf-8"))
except Exception:
    la = {}
print(f"8B direct 已完成 {len(d)} 条 | 8B som {len(s)} 条 | LA 探针 {len(la)} 条\n")

for k in "000512_002 000512_004 000085_001 000085_002 000246_001 000246_012 000001_006".split():
    print(f"=== {k} ===")
    dd = d.get(k)
    if dd is None:
        print("  8B direct: 尚未处理")
    else:
        print(f"  8B direct bbox={dd.get('bbox')} n={dd.get('n')}")
    if k in s:
        print(f"  8B som: n_cands={s[k].get('n_cands')} num={s[k].get('num')} bbox={s[k].get('bbox')}")
    if k in la:
        v = la[k]
        print(f"  LA: n_boxes={v['n_boxes']} 前3={v['boxes'][:3]}")
        if dd and dd.get("bbox"):
            print(f"  IoU(LA.first, 8B.direct) = {iou(v['boxes'][0], dd['bbox']):.3f}")
            print(f"  IoU(LA.best , 8B.direct) = "
                  f"{max(iou(b, dd['bbox']) for b in v['boxes']):.3f}")
        if s.get(k, {}).get("bbox"):
            print(f"  IoU(LA.first, 8B.som) = {iou(v['boxes'][0], s[k]['bbox']):.3f}")
            print(f"  IoU(LA.best , 8B.som) = "
                  f"{max(iou(b, s[k]['bbox']) for b in v['boxes']):.3f}")
    print()
PY
