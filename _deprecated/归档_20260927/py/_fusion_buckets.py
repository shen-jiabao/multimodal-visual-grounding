# -*- coding: utf-8 -*-
"""统计跨家族一致性：决定「加权平均」与「CLIP 仲裁」各自覆盖多少样本。"""
import json
from pathlib import Path

d = json.loads(Path("/root/output/fusion_detail.json").read_text(encoding="utf-8"))


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


cross_hi = cross_mid = cross_lo = 0
pair_hi = {"L2-M": 0, "L2-G": 0, "M-G": 0}
for k, v in d.items():
    bx = v["boxes"]
    L2, M, G = bx["L2_locate_v2_hires"], bx["M_gd3modal"], bx["G_groundingdino"]
    im, ig, mg = iou(L2, M), iou(L2, G), iou(M, G)
    best = max(im, ig, mg)
    if im >= 0.5: pair_hi["L2-M"] += 1
    if ig >= 0.5: pair_hi["L2-G"] += 1
    if mg >= 0.5: pair_hi["M-G"] += 1
    if best >= 0.5:
        cross_hi += 1
    elif best >= 0.3:
        cross_mid += 1
    else:
        cross_lo += 1

n = len(d)
print(f"总样本 {n}")
print(f"跨家族 max IoU>=0.5 (可加权平均): {cross_hi} ({100*cross_hi/n:.1f}%)")
print(f"跨家族 max IoU 0.3~0.5        : {cross_mid} ({100*cross_mid/n:.1f}%)")
print(f"跨家族 max IoU <0.3 (需仲裁)   : {cross_lo} ({100*cross_lo/n:.1f}%)")
print("高重合的家族对计数:", pair_hi)
