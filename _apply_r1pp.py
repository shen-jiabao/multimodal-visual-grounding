# -*- coding: utf-8 -*-
"""采纳 R1″：纯空间词(非序数) 且 基线面积>0.35 的样本 → 采用 Locate 框。
输入: fusion_g4_r2r1p.json（当前锁定版）
输出: fusion_g4_r2r1p_r1pp.json
"""
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


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = area(a)+area(b)-it
    return it/ua if ua > 1e-9 else 0.


def norm(b):
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0., min(1., x1)), max(0., min(1., y1))
    x2, y2 = max(0., min(1., x2)), max(0., min(1., y2))
    return [x1, y1, min(1., max(x2, x1+1e-3)), min(1., max(y2, y1+1e-3))]


cand = [k for k in cur
        if (not ORD.search(Q[k]["query"])) and SPAT.search(Q[k]["query"])
        and area(base[k]["bbox"]) > 0.35]

out = {}
applied = 0
for k, rec in cur.items():
    entry = dict(rec)
    if k in cand:
        nb = norm(l2[k]["bbox"])
        if iou(nb, rec["bbox"]) < 0.999:
            entry["bbox"] = [round(float(v), 6) for v in nb]
            applied += 1
    out[k] = entry

p = R / "fusion_g4_r2r1p_r1pp.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

chg = [k for k in out if iou(out[k]["bbox"], cur[k]["bbox"]) < 0.999]
chg_base = [k for k in out if iou(out[k]["bbox"], base[k]["bbox"]) < 0.999]
CONF = ["000282_004", "000282_008", "002004_001", "002004_002"]
print(f"R1″ 候选 {len(cand)}；实际改动 {applied}")
print(f"相对当前锁定版改变 {len(chg)}；相对基线总改动 {len(chg_base)} "
      f"({100*len(chg_base)/len(out):.1f}%)")
print(f"大框: 基线 {sum(1 for v in base.values() if area(v['bbox'])>0.35)} → "
      f"锁定版 {sum(1 for v in cur.values() if area(v['bbox'])>0.35)} → "
      f"新版 {sum(1 for v in out.values() if area(v['bbox'])>0.35)}")
print(f"护栏样本是否被改: {[(k, k in chg) for k in CONF]}")
print(f"输出: {p}")
