# -*- coding: utf-8 -*-
"""本地校验：保守替换(885) 与 保守替换+面积护栏(866) 两版结果。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
files = {
    "无护栏(885)": R / "fusion_conservative.json",
    "有护栏(866)": R / "fusion_conservative_guard.json",
}
CONF = {
    "000282_004": [0.2021, 0.3015, 0.3054, 0.4004],
    "000282_008": [0.0312, 0.3244, 0.1368, 0.5068],
    "002004_001": [0.4208, 0.2913, 0.4667, 0.4365],
    "002004_002": [0.4543, 0.5115, 0.5251, 0.6628],
}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    it = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


FIELDS = {"visible", "infrared", "depth", "query", "bbox"}
for name, p in files.items():
    d = json.loads(p.read_text(encoding="utf-8"))
    print("=" * 74)
    print(f"[{name}] {p.name}")
    print(f"  条目: {len(d)}  (基线 {len(base)})")
    chg = [k for k in d if any(abs(a-b) > 1e-6 for a, b in zip(d[k]["bbox"], base[k]["bbox"]))]
    print(f"  与基线不同: {len(chg)}")
    miss = [k for k, v in d.items() if set(v) != FIELDS]
    bad = [k for k, v in d.items()
           if not (0.0 <= v["bbox"][0] < v["bbox"][2] <= 1.0
                   and 0.0 <= v["bbox"][1] < v["bbox"][3] <= 1.0)]
    print(f"  字段异常: {len(miss)}  非法bbox: {len(bad)}")
    big = sum(1 for v in d.values() if area(v["bbox"]) > 0.35)
    print(f"  大框(>0.35): {big}  (基线 385)")
    for k, gt in CONF.items():
        print(f"  {k}: 改动={'是' if k in chg else '否'}  IoU={iou(d[k]['bbox'], gt):.4f}")
