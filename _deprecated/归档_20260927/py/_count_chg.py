# -*- coding: utf-8 -*-
"""统计 G4+R2+R1′ 相对基线的总改动面。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
new = json.loads((R / "fusion_g4_r2r1p.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = area(a)+area(b)-it
    return it/ua if ua > 1e-9 else 0.


chg = [k for k in new if iou(new[k]["bbox"], base[k]["bbox"]) < 0.999]
big0 = sum(1 for v in base.values() if area(v["bbox"]) > 0.35)
big1 = sum(1 for v in new.values() if area(v["bbox"]) > 0.35)
print(f"相对基线改动: {len(chg)} ({100*len(chg)/len(new):.1f}%)")
print(f"大框: 基线 {big0} → 新 {big1}")
