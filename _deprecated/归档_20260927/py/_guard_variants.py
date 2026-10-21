# -*- coding: utf-8 -*-
"""分析 000247_008 的融合细节，并量化几种更严护栏的影响。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
final = json.loads((R / "result_gd_3modal_final.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
g = json.loads((R / "predictions_grounding.json").read_text(encoding="utf-8"))
Q = json.loads((R / "初赛数据集-基于大模型的多模态视觉理解与推理"
                / "queries" / "queries.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    it = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


k = "000247_008"
print("=== 000247_008 细节 ===")
print("query:", Q[k]["query"])
for name, d, tag in [(("基线 三模态GD(RGB+IR+Depth)"), base, "M"),
                     ("Locate-v2 (纯RGB)", l2, "L2"),
                     ("GroundingDINO (纯RGB)", g, "G"),
                     ("最终提交(0.6L2+0.4G)", final, "F")]:
    b = d[k]["bbox"]
    print(f"  {name:28s} bbox={[round(v,3) for v in b]}  area={area(b):.3f}")
print(f"  IoU(L2,G)={iou(l2[k]['bbox'], g[k]['bbox']):.3f}  "
      f"IoU(M,L2)={iou(base[k]['bbox'], l2[k]['bbox']):.3f}  "
      f"IoU(M,G)={iou(base[k]['bbox'], g[k]['bbox']):.3f}")

# 几种护栏的影响
c885 = json.loads((R / "fusion_conservative.json").read_text(encoding="utf-8"))
c885_changed = [kk for kk in c885 if any(abs(a-b) > 1e-6 for a, b in
                                         zip(c885[kk]["bbox"], base[kk]["bbox"]))]
candidates = {}   # k -> fused box
for kk in c885_changed:
    lb, gb = l2[kk]["bbox"], g[kk]["bbox"]
    candidates[kk] = [0.6*lb[i] + 0.4*gb[i] for i in range(4)]


def evaluate(name, keep_fn):
    rep = [kk for kk, fb in candidates.items() if keep_fn(fb, base[kk]["bbox"])]
    big0 = sum(1 for v in base.values() if area(v["bbox"]) > 0.35)
    bigf = sum(1 for kk in base
               if area(candidates[kk] if kk in rep else base[kk]["bbox"]) > 0.35)
    fixed = sum(1 for kk in rep
                if area(base[kk]["bbox"]) > 0.35 and area(candidates[kk]) <= 0.35)
    bk = "是" if k in rep else "否"
    print(f"  {name:34s} 替换={len(rep):4d}  大框={bigf:3d}(基线{big0})  "
          f"修大框={fixed:3d}  本例替换={bk}")


print("\n=== 护栏变体影响 ===")
evaluate("G1 当前(替换框≤0.5)", lambda fb, mb: area(fb) <= 0.5)
evaluate("G2 替换框≤0.35", lambda fb, mb: area(fb) <= 0.35)
evaluate("G3 不允许放大(≤1.25×基线)", lambda fb, mb: area(fb) <= 1.25*area(mb))
evaluate("G4 同时≤0.35 且 ≤1.25×基线",
         lambda fb, mb: area(fb) <= min(0.35, 1.25*area(mb)))
evaluate("G5 仅当基线也偏大(>0.15)才替换",
         lambda fb, mb: area(mb) > 0.15 and area(fb) <= 1.25*area(mb))
