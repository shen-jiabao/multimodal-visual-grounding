# -*- coding: utf-8 -*-
"""R1″ 只读抽检：纯空间词(非序数) 且 基线面积>0.35 的 175 条中抽 10 条出图。
不改动任何提交文件；输出 viz_out/review_v2/
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "review_v2"
VIZ.mkdir(parents=True, exist_ok=True)

base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
new = json.loads((R / "fusion_g4_r2r1p.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
g = json.loads((R / "predictions_grounding.json").read_text(encoding="utf-8"))
Q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))

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


# R1″ 候选集合
cand = [k for k in new
        if (not ORD.search(Q[k]["query"])) and SPAT.search(Q[k]["query"])
        and area(base[k]["bbox"]) > 0.35]
cand.sort(key=lambda k: area(base[k]["bbox"]), reverse=True)

# 大框影响量（若采纳 R1″）
def big_count(mapping):
    return sum(1 for k in new if area(mapping(k)) > 0.35)

cur_big = big_count(lambda k: new[k]["bbox"])
r1pp_big = big_count(lambda k: l2[k]["bbox"] if k in cand else new[k]["bbox"])
print(f"R1″ 候选 {len(cand)} 条（纯空间词 且 基线面积>0.35）")
print(f"大框: 当前提交 {cur_big} → 若采纳 R1″ {r1pp_big}")
print(f"护栏样本是否在候选中: {[k for k in ['000282_004','000282_008','002004_001','002004_002'] if k in cand]}")

# 抽样：确保含 000023_006，其余按场景去重、基线面积从大到小
picks, seen = [], set()
if "000023_006" in cand:
    picks.append("000023_006"); seen.add("000023")
for k in cand:
    if len(picks) >= 10:
        break
    sc = k.split("_")[0]
    if sc in seen:
        continue
    seen.add(sc)
    picks.append(k)

COLORS = {"new": (0, 0, 255), "L2": (255, 120, 0), "M": (150, 150, 150),
          "G": (200, 0, 200)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


print(f"\n{'qid':14s} {'基线M':>7s} {'L2':>7s} {'G':>7s} {'当前提交':>7s}  query")
for qid in picks:
    img = imread_u(DATA / Q[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H*1000/W)
    img = cv2.resize(img, (dw, dh))
    for name, b, th in [("G", g[qid]["bbox"], 2), ("M", base[qid]["bbox"], 2),
                        ("L2", l2[qid]["bbox"], 3), ("new", new[qid]["bbox"], 5)]:
        c = COLORS[name]
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, th)
        if name in ("new", "L2"):
            cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cap = np.full((102, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[R1\" 候选] {qid}: {Q[qid]['query'][:100]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"area  M(base)={area(base[qid]['bbox']):.3f}  L2={area(l2[qid]['bbox']):.3f}  "
                     f"G={area(g[qid]['bbox']):.3f}  当前提交={area(new[qid]['bbox']):.3f}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(cap, "红粗=当前提交  蓝粗=R1″建议改用的Locate框  灰细=M(base,非真值)  品红细=G",
                (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"R1pp_{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"{qid:14s} {area(base[qid]['bbox']):7.3f} {area(l2[qid]['bbox']):7.3f} "
          f"{area(g[qid]['bbox']):7.3f} {area(new[qid]['bbox']):7.3f}  {Q[qid]['query'][:44]}")
print(f"\n输出目录: {VIZ}")
