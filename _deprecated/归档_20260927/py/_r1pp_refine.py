# -*- coding: utf-8 -*-
"""R1″ 真实分档 + 更保守口径量化（按 L2 框面积再筛一层）。"""
import json
import re
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
cur = json.loads((R / "fusion_g4_r2r1p.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
Q = json.loads((R / "初赛数据集-基于大模型的多模态视觉理解与推理"
                / "queries" / "queries.json").read_text(encoding="utf-8"))
ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                  r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                  r"beside|next to|between|foreground|background|upper|lower", re.I)


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


cand = [k for k in cur
        if (not ORD.search(Q[k]["query"])) and SPAT.search(Q[k]["query"])
        and area(base[k]["bbox"]) > 0.35]

print("=== 基线面积分档（R1″ 175 条）===")
for lo, hi in [(0.9, 1.01), (0.8, 0.9), (0.6, 0.8), (0.45, 0.6), (0.35, 0.45)]:
    n = sum(1 for k in cand if lo <= area(base[k]["bbox"]) < hi)
    print(f"  基线 [{lo},{hi}): {n}")

print("\n=== 按 L2 框面积再筛 ===")
small = [k for k in cand if area(l2[k]["bbox"]) <= 0.35]
bigL2 = [k for k in cand if area(l2[k]["bbox"]) > 0.35]
print(f"  L2 ≤0.35（物体尺度）: {len(small)}")
print(f"  L2 >0.35（同样是大区域）: {len(bigL2)}")
print("  后者清单（建议保留基线）:")
for k in sorted(bigL2, key=lambda k: -area(l2[k]["bbox"])):
    print(f"    {k}  M={area(base[k]['bbox']):.3f} L2={area(l2[k]['bbox']):.3f}  {Q[k]['query'][:56]}")

print(f"\n若采用「基线>0.35 且 L2≤0.35」→ 采纳 {len(small)} 条，"
      f"回退 {len(bigL2)} 条；大框数: "
      f"{sum(1 for k in cur if area(l2[k]['bbox'] if k in small else cur[k]['bbox'])>0.35)}")
