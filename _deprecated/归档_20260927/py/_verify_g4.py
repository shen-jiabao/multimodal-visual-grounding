# -*- coding: utf-8 -*-
"""G4 定稿本地复核：条目/字段/合法性/确认样本/无人机/改动面。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
g4 = json.loads((R / "fusion_g4.json").read_text(encoding="utf-8"))
test = json.loads((R / "初赛数据集-基于大模型的多模态视觉理解与推理"
                   / "test.json").read_text(encoding="utf-8"))
CONF = {
    "000282_004": [0.2021, 0.3015, 0.3054, 0.4004],
    "000282_008": [0.0312, 0.3244, 0.1368, 0.5068],
    "002004_001": [0.4208, 0.2913, 0.4667, 0.4365],
    "002004_002": [0.4543, 0.5115, 0.5251, 0.6628],
}
FIELDS = {"visible", "infrared", "depth", "query", "bbox"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    it = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


print(f"条目: {len(g4)} (基线 {len(base)}); key 缺失 {len(set(test)-set(g4))}, "
      f"多余 {len(set(g4)-set(test))}")
bad_field = [k for k, v in g4.items() if set(v) != FIELDS]
bad_box = [k for k, v in g4.items()
           if not (0.0 <= v["bbox"][0] < v["bbox"][2] <= 1.0
                   and 0.0 <= v["bbox"][1] < v["bbox"][3] <= 1.0)]
print(f"字段异常: {len(bad_field)}; 非法 bbox: {len(bad_box)}")

chg = [k for k in g4 if any(abs(a-b) > 1e-6 for a, b in
                            zip(g4[k]["bbox"], base[k]["bbox"]))]
print(f"与基线不同: {len(chg)} (期望 432)")
big0 = sum(1 for v in base.values() if area(v["bbox"]) > 0.35)
big4 = sum(1 for v in g4.values() if area(v["bbox"]) > 0.35)
print(f"大框(>0.35): {big0} → {big4} (期望 348)")

print("\n4 个已确认样本:")
for k, gt in CONF.items():
    print(f"  {k}: 改动={'是' if k in chg else '否'}  IoU={iou(g4[k]['bbox'], gt):.4f}")

k = "000247_008"
print(f"\n无人机样本 {k}: bbox={[round(v,3) for v in g4[k]['bbox']]} "
      f"area={area(g4[k]['bbox']):.4f}  改动={'是' if k in chg else '否'}")
