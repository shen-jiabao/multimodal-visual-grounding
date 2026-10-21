# -*- coding: utf-8 -*-
"""R1″ 保留基线的 47 条：逐条出图 + 汇总看板(3 张)。
输出 viz_out/review_v4_all/
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "review_v4_all"
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
kept = [k for k in cand if area(l2[k]["bbox"]) > 0.35]
kept.sort(key=lambda k: -area(base[k]["bbox"]))
print(f"保留基线 {len(kept)} 条")


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


CELL_W = 470


def make_cell(qid):
    img = imread_u(DATA / Q[qid]["visible"])
    H, W = img.shape[:2]
    ch = int(H*CELL_W/W)
    img = cv2.resize(img, (CELL_W, ch))
    for b, c, th in [(base[qid]["bbox"], (150, 150, 150), 2),
                     (l2[qid]["bbox"], (255, 120, 0), 2),
                     (new[qid]["bbox"], (0, 0, 255), 3)]:
        p1 = (int(b[0]*CELL_W), int(b[1]*ch)); p2 = (int(b[2]*CELL_W), int(b[3]*ch))
        cv2.rectangle(img, p1, p2, c, th)
    cap = np.full((46, CELL_W, 3), 245, np.uint8)
    cv2.putText(cap, f"{qid}  M={area(base[qid]['bbox']):.2f} L2={area(l2[qid]['bbox']):.2f}",
                (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, Q[qid]["query"][:58], (6, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)
    cell = np.vstack([img, cap])
    cv2.imencode(".png", cell)[1].tofile(str(VIZ / f"{qid}.png"))
    return cell


cells = [make_cell(k) for k in kept]
print(f"逐条图已保存 {len(cells)} 张 -> {VIZ}")

# 汇总看板：4 列
COLS = 4
for si in range(0, len(cells), 20):
    chunk = cells[si:si+20]
    rows = []
    for i in range(0, len(chunk), COLS):
        row = chunk[i:i+COLS]
        hmax = max(c.shape[0] for c in row)
        row = [np.vstack([c, np.full((hmax-c.shape[0], CELL_W, 3), 255, np.uint8)])
               if c.shape[0] < hmax else c for c in row]
        while len(row) < COLS:
            row.append(np.full((hmax, CELL_W, 3), 255, np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    # 顶部标题条
    head = np.full((34, sheet.shape[1], 3), 240, np.uint8)
    cv2.putText(head, f"R1'' kept-baseline samples  sheet {si//20+1}  "
                      f"({len(chunk)} items)  red=submitted  blue=Locate(not used)  gray=M(base)",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.vstack([head, sheet])
    out = VIZ / f"_sheet{si//20+1}.png"
    cv2.imencode(".png", sheet)[1].tofile(str(out))
    print(f"看板已保存 {out}  尺寸 {sheet.shape[1]}x{sheet.shape[0]}")
