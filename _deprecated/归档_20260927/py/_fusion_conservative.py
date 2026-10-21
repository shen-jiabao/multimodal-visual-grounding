# -*- coding: utf-8 -*-
"""保守替换：仅当 L2≈G(IoU>=0.5) 且基线 M 与两者都不一致(IoU<0.5) 时，
用 0.6*L2 + 0.4*G 替换 bbox；其余沿用基线 result_gd_3modal_v2.json。
输出 /root/output/fusion_conservative.json
"""
import json
from pathlib import Path

OUT = Path("/root/output")
L2 = json.load(open(OUT / "predictions_v2_highres_slow.json"))
M = json.load(open(OUT / "result_gd_3modal_v2.json"))
G = json.load(open(OUT / "predictions_grounding.json"))

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
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def norm(b):
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    if x2 - x1 < 1e-3:
        x2 = min(1.0, x1 + 1e-3)
    if y2 - y1 < 1e-3:
        y2 = min(1.0, y1 + 1e-3)
    return [x1, y1, x2, y2]


out, replaced = {}, []
for k, rec in M.items():
    entry = dict(rec)
    if k in L2 and k in G:
        l2, m, g = L2[k]["bbox"], rec["bbox"], G[k]["bbox"]
        if iou(l2, g) >= 0.5 and iou(l2, m) < 0.5 and iou(g, m) < 0.5:
            b = norm([0.6*l2[i] + 0.4*g[i] for i in range(4)])
            entry["bbox"] = [round(float(v), 6) for v in b]
            replaced.append(k)
    out[k] = entry

p = OUT / "fusion_conservative.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"总条目 {len(out)}")
print(f"替换数 {len(replaced)}  ({100*len(replaced)/len(out):.2f}%)")
print(f"未改动 {len(out)-len(replaced)}")

big_m = [k for k, r in M.items() if area(r["bbox"]) > 0.35]
big_o = [k for k, r in out.items() if area(r["bbox"]) > 0.35]
print(f"\n大框(>0.35): 基线 {len(big_m)} → 保守后 {len(big_o)}  "
      f"(修掉 {len(set(big_m)-set(big_o))}, 新增 {len(set(big_o)-set(big_m))})")

print("\n4 个已确认样本:")
for k, gt in CONF.items():
    mb = M[k]["bbox"]
    ob = out[k]["bbox"]
    chg = "改动" if any(abs(a-b) > 1e-6 for a, b in zip(mb, ob)) else "未动"
    print(f"  {k}: {chg}  基线IoU={iou(mb,gt):.4f} → 提交IoU={iou(ob,gt):.4f}")

# 自检：替换集合是否与预期口径一致
print(f"\n替换集合前 5 个: {replaced[:5]}")
print(f"输出文件: {p}  {p.stat().st_size/1024:.0f} KB")
