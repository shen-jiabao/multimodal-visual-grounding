# -*- coding: utf-8 -*-
"""核对 R1″ 的 Tier 构成：Tier A(基线≈近整图 >0.8) / Tier B(0.35~0.8)，并标出已人工确认样本。"""
import json
import re
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
cur = json.loads((R / "fusion_g4_r2r1p.json").read_text(encoding="utf-8"))
new = json.loads((R / "fusion_g4_r2r1p_r1pp.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
Q = json.loads((R / "初赛数据集-基于大模型的多模态视觉理解与推理"
                / "queries" / "queries.json").read_text(encoding="utf-8"))

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                  r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                  r"beside|next to|between|foreground|background|upper|lower", re.I)


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = area(a)+area(b)-it
    return it/ua if ua > 1e-9 else 0.


cand = [k for k in cur
        if (not ORD.search(Q[k]["query"])) and SPAT.search(Q[k]["query"])
        and area(base[k]["bbox"]) > 0.35]
verified = ["000023_006"]          # 你人工判定"蓝色框正确"的 Tier B 样本
tierA = [k for k in cand if area(base[k]["bbox"]) > 0.8]
tierB = [k for k in cand if area(base[k]["bbox"]) <= 0.8]
print(f"R1″ 候选 {len(cand)} = TierA(基线>0.8) {len(tierA)} + TierB(0.35~0.8) {len(tierB)}")
print(f"TierB 中已人工确认: {[k for k in verified if k in tierB]}")
print(f"TierB 未验证: {len([k for k in tierB if k not in verified])}")

# 保守口径：TierA + 已确认
safe = tierA + [k for k in verified if k in tierB]
out_kept = []
for k in cand:
    if k not in safe:
        out_kept.append((k, round(area(base[k]['bbox']), 3), round(area(l2[k]['bbox']), 3),
                         Q[k]['query'][:52]))
print(f"\n若收缩为 TierA+已确认({len(safe)}) → 需回退 {len(out_kept)} 条:")
for k, ab, al, q in sorted(out_kept, key=lambda x: -x[1])[:30]:
    print(f"  {k}  M={ab:.3f} L2={al:.3f}  {q}")
