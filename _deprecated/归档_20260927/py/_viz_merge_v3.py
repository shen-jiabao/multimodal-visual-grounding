# -*- coding: utf-8 -*-
"""生成 v3 结构版专属抽检：输出到 viz_out/locate_merge_v3_structure/。

红=融合(Locate) 绿=基线(仅二者不同时显示)。固定 seed 保证可复现。
"""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "locate_merge_v3_structure"
VIZ.mkdir(parents=True, exist_ok=True)
# 清空旧文件，保证这是纯 v3 集
for old in VIZ.glob("*.png"):
    old.unlink()

final = json.loads((R / "result_locate_merge.json").read_text(encoding="utf-8"))
base = json.loads((R / "result_gd_3modal_fullseq.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    it = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / ua if ua > 1e-9 else 0.0


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


keys = sorted(final)
changed = [k for k in keys if final[k]["bbox"] != base[k]["bbox"]]
same = [k for k in keys if final[k]["bbox"] == base[k]["bbox"]]
random.seed(42)
picks = [
    "000023_002", "000023_004", "001038_003", "016774_001",   # 序数/空间关键
    "012161_004",                                               # 结构护栏修复点
    "000282_008",                                               # 护栏样本
] + random.sample(changed, 6) + random.sample(same, 3)
picks = list(dict.fromkeys(picks))[:14]

RED = (0, 0, 255)
GREEN = (0, 200, 0)
made = 0
for qid in picks:
    img = imread_u(DATA / queries[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H * 1000 / W)
    img = cv2.resize(img, (dw, dh))
    fb, bb = final[qid]["bbox"], base[qid]["bbox"]
    changed_flag = fb != bb
    if changed_flag:
        p1 = (int(bb[0] * dw), int(bb[1] * dh))
        p2 = (int(bb[2] * dw), int(bb[3] * dh))
        cv2.rectangle(img, p1, p2, GREEN, 2)
        cv2.putText(img, "base", (p1[0], max(16, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
    p1 = (int(fb[0] * dw), int(fb[1] * dh))
    p2 = (int(fb[2] * dw), int(fb[3] * dh))
    cv2.rectangle(img, p1, p2, RED, 4)
    cv2.putText(img, "merge", (max(0, p1[0]), max(16, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
    cap = np.full((70, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[v3] {qid} | IoU(merge,base)={iou(fb, bb):.2f} "
                     f"{'<== CHANGED' if changed_flag else '(同基线)'}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, queries[qid]["query"][:95], (8, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(cap, "red=merge(Locate) green=base", (8, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    made += 1
print(f"v3 专属抽检 {made} 张 -> {VIZ}")
for k in picks:
    print("  ", k, "changed=", final[k]["bbox"] != base[k]["bbox"])