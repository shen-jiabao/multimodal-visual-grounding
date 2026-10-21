# -*- coding: utf-8 -*-
"""A 方案融合结果校验：① 4 个已人工确认样本是否保住；② 大框(>0.35)数量变化。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
fus = json.loads((R / "fusion_consensus.json").read_text(encoding="utf-8"))
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


CONFIRMED = {
    "000282_004": [0.2021, 0.3015, 0.3054, 0.4004],
    "000282_008": [0.0312, 0.3244, 0.1368, 0.5068],
    "002004_001": [0.4208, 0.2913, 0.4667, 0.4365],
    "002004_002": [0.4543, 0.5115, 0.5251, 0.6628],
}
print("=== 已确认样本在融合前后的表现（IoU 对人工确认框）===")
for k, gt in CONFIRMED.items():
    mb = base[k]["bbox"] if isinstance(base[k], dict) else base[k]
    fb = fus[k]["bbox"] if isinstance(fus[k], dict) else fus[k]
    print(f"{k}: 基线={[round(v,3) for v in mb]} IoU={iou(mb,gt):.3f} | "
          f"融合={[round(v,3) for v in fb]} IoU={iou(fb,gt):.3f}")

def getb(v):
    return v["bbox"] if isinstance(v, dict) else v

big_base = [k for k, v in base.items() if area(getb(v)) > 0.35]
big_fus = [k for k, v in fus.items() if area(getb(v)) > 0.35]
print(f"\n大框(面积>0.35): 基线 {len(big_base)} → 融合 {len(big_fus)}")
fixed = [k for k in big_base if k not in big_fus]
broke = [k for k in big_fus if k not in big_base]
print(f"  融合后修掉的大框: {len(fixed)}；新增大框: {len(broke)}")
print("  修掉示例:", fixed[:8])
print("  新增示例:", broke[:8])
