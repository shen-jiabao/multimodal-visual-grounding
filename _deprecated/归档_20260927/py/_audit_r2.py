# -*- coding: utf-8 -*-
"""抽查融合 R2 回退样本：拆出放大漂移回退的样本，人工可判的主要模式。"""
import json
import re
from pathlib import Path

R = Path(__file__).resolve().parent
LOC = json.loads((R / "locate_bbox_all.json").read_text(encoding="utf-8"))
BASE = json.loads((R / "result_gd_3modal_fullseq.json").read_text(encoding="utf-8"))
Q = json.loads((R / "初赛数据集-基于大模型的多模态视觉理解与推理"
                / "queries" / "queries.json").read_text(encoding="utf-8"))

AREA_LIMIT = 0.35
IoU_LOW = 0.2
REGION_RE = re.compile(
    r"\b(lake|water|pond|river|sea|ocean|sky|cloud|grass|lawn|ground|floor|"
    r"road|street|sidewalk|path|field|sand|snow|wall|fence|barrier|"
    r"bamboo|tree|door|area|region|scenery|background|foreground|landscape)\b",
    re.I)


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


# 统计 R2 放大漂移池（与融合脚本同规则）
r2 = []
r1 = []
r0 = []
for k, rec in BASE.items():
    lb = LOC.get(k, {}).get("bbox")
    bb = rec["bbox"]
    q = Q[k]["query"]
    is_region = bool(REGION_RE.search(q))
    if lb is None:
        r0.append(k)
    elif area(lb) > AREA_LIMIT and not is_region:
        r1.append((k, area(lb), area(bb)))
    elif iou(lb, bb) < IoU_LOW and not is_region and area(lb) > 1.5 * area(bb):
        r2.append((k, round(iou(lb, bb), 3), round(area(lb), 4), round(area(bb), 4)))

print(f"R0(无框回退) {len(r0)} | R1(面积>0.35非区域) {len(r1)} | R2(放大漂移非区域) {len(r2)}")
print("\n== R2 放大最大 15 条 ==")
for k, i, al, ab in sorted(r2, key=lambda x: -x[2])[:15]:
    print(f"  {k} IoU={i}  loc_area={al}  base_area={ab}  | {Q[k]['query'][:60]}")
print("\n== R2 放大倍数最大 10 条 ==")
for k, i, al, ab in sorted(r2, key=lambda x: -(x[2]/max(x[3],1e-6)))[:10]:
    print(f"  {k} IoU={i}  loc/base={al/ab:.1f}x  | {Q[k]['query'][:60]}")