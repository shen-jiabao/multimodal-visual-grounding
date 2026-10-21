# -*- coding: utf-8 -*-
"""回归点检集 v0（12 条，只读）：
 4 条已确认护栏 + 4 条 G4 替换(分层) + 4 条空间/序数类未替换(基线偏大)
图注中性：M(base) 不预设真值。输出 viz_out/review_v0/
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "review_v0"
VIZ.mkdir(parents=True, exist_ok=True)

base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
final = json.loads((R / "fusion_g4.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
g = json.loads((R / "predictions_grounding.json").read_text(encoding="utf-8"))
Q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))

ORDS = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b|"
                  r"\b(two|three|four|five|six|seven|eight|nine|ten|pair|both)\b|"
                  r"from (the )?(left|right) to", re.I)
SPAT = re.compile(r"leftmost|rightmost|topmost|bottommost|nearest|closest|farthest|furthest|"
                  r"\bleft\b|\bright\b|\btop\b|\bbottom\b|behind|in front|above|below|"
                  r"beside|next to|between|foreground|background|upper|lower", re.I)


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    it = iw*ih
    ua = area(a) + area(b) - it
    return it/ua if ua > 1e-9 else 0.0


keys = sorted(final)
replaced = [k for k in keys
            if any(abs(a-b) > 1e-6 for a, b in zip(final[k]["bbox"], base[k]["bbox"]))]
not_replaced = [k for k in keys if k not in replaced]


def qtype(k):
    q = Q[k]["query"]
    if ORDS.search(q):
        return "序数"
    if SPAT.search(q):
        return "空间"
    return "普通"


CONF = ["000282_004", "000282_008", "002004_001", "002004_002"]
picks = [("护栏-已确认", k) for k in CONF]

# 替换组分层：面积缩小/放大(≤1.25x) × 类型
def bucket(k):
    ab, af = area(base[k]["bbox"]), area(final[k]["bbox"])
    return "缩小" if af < ab*0.98 else ("放大" if af > ab*1.02 else "持平")

want = [("空间", "缩小"), ("空间", "放大"), ("序数", "缩小"), ("普通", "缩小")]
used = set(CONF)
for t, b in want:
    for k in replaced:
        if k in used:
            continue
        if qtype(k) == t and bucket(k) == b:
            picks.append((f"替换-{t}类-面积{b}", k))
            used.add(k)
            break

# 空间/序数类、基线偏大、未被替换
for k in not_replaced:
    if len([p for p in picks if p[0].startswith("未替换")]) >= 4:
        break
    if qtype(k) in ("空间", "序数") and area(base[k]["bbox"]) > 0.20:
        picks.append((f"未替换-{qtype(k)}类-基线偏大", k))

COLORS = {"final": (0, 0, 255), "M": (150, 150, 150),
          "L2": (255, 120, 0), "G": (200, 0, 200)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


print(f"{'qid':14s} {'分组':22s} {'替换':4s} {'L2':>7s} {'M':>7s} {'G':>7s} {'final':>7s}  query")
for tag, k in picks:
    img = imread_u(DATA / Q[k]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H * 1000 / W)
    img = cv2.resize(img, (dw, dh))
    bx = {"G": g[k]["bbox"], "L2": l2[k]["bbox"],
          "M": base[k]["bbox"], "final": final[k]["bbox"]}
    for name, th in [("G", 2), ("L2", 2), ("M", 2), ("final", 5)]:
        b = bx[name]
        c = COLORS[name]
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, th)
        if name in ("final", "M"):
            cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    rep = any(abs(a-b) > 1e-6 for a, b in zip(final[k]["bbox"], base[k]["bbox"]))
    cap = np.full((102, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[{tag}] {k}: {Q[k]['query'][:100]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"G4替换: {'是' if rep else '否'}   "
                     f"IoU(final,M)={iou(final[k]['bbox'], base[k]['bbox']):.3f}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 200), 1, cv2.LINE_AA)
    cv2.putText(cap, f"area  L2={area(bx['L2']):.3f}  M(base)={area(bx['M']):.3f}  "
                     f"G={area(bx['G']):.3f}  final={area(bx['final']):.3f}",
                (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(cap, "红粗=final  灰细=M(base)  蓝细=Locate  品红细=G   (M(base) 非真值)",
                (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"{tag}_{k}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"{k:14s} {tag:22s} {'是' if rep else '否':4s} "
          f"{area(bx['L2']):7.3f} {area(bx['M']):7.3f} {area(bx['G']):7.3f} "
          f"{area(bx['final']):7.3f}  {Q[k]['query'][:44]}")
print(f"\n共 {len(picks)} 条，输出目录: {VIZ}")
