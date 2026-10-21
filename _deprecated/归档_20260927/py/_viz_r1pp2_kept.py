# -*- coding: utf-8 -*-
"""R1″ 精炼版中「保留基线」的 47 条示例出图（供你后续逐条判）。
输出 viz_out/review_v4/  （红粗=当前提交 蓝细=Locate 灰细=M(base)）
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "review_v4"
VIZ.mkdir(parents=True, exist_ok=True)

base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
new = json.loads((R / "fusion_r1pp2.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
Q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                  r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                  r"beside|next to|between|foreground|background|upper|lower", re.I)


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


cand = [k for k in new
        if (not ORD.search(Q[k]["query"])) and SPAT.search(Q[k]["query"])
        and area(base[k]["bbox"]) > 0.35]
kept = [k for k in cand if area(l2[k]["bbox"]) > 0.35]      # 保留基线的 47 条
picks = ["016606_001", "002137_002", "015248_003", "009823_006",
         "016130_002", "008731_003"]
picks = [k for k in picks if k in kept] or kept[:6]

COLORS = {"new": (0, 0, 255), "L2": (255, 120, 0), "M": (150, 150, 150)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


print(f"保留基线共 {len(kept)} 条；本次出图 {len(picks)} 条")
for qid in picks:
    img = imread_u(DATA / Q[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H*1000/W)
    img = cv2.resize(img, (dw, dh))
    for name, b, th in [("M", base[qid]["bbox"], 2), ("L2", l2[qid]["bbox"], 2),
                        ("new", new[qid]["bbox"], 5)]:
        c = COLORS[name]
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, th)
        if name in ("new", "L2"):
            cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cap = np.full((102, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[R1″保留基线] {qid}: {Q[qid]['query'][:100]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"area  M(base)={area(base[qid]['bbox']):.3f}  "
                     f"L2={area(l2[qid]['bbox']):.3f}  当前提交={area(new[qid]['bbox']):.3f}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(cap, "红粗=当前提交(保留基线)  蓝细=Locate(未采纳)  灰细=M(base)",
                (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"kept_{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"已保存 {out}  M={area(base[qid]['bbox']):.3f} L2={area(l2[qid]['bbox']):.3f}")
