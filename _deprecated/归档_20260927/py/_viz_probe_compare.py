# -*- coding: utf-8 -*-
"""将探针结果与基线叠画对比，输出可视化供人工判定。

红粗=Locate 探针；绿细=基线 fullseq。含问句与 IoU。
输出 viz_out/locate_probe/{qid}.png
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "locate_probe"
VIZ.mkdir(parents=True, exist_ok=True)

probe = json.loads((R / "probe_result.json").read_text(encoding="utf-8"))
base = json.loads((R / "result_gd_3modal_fullseq.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    it = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / ua if ua > 1e-9 else 0.0


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


# 各图取 IoU<0.5 的分歧样本（每图最多 5 张）
by_img = defaultdict(list)
for k, v in probe.items():
    boxes = v.get("boxes") or []
    loc = boxes[0] if boxes else None
    iv = iou(loc, base[k]["bbox"]) if loc else 0.0
    if loc and iv < 0.5:
        by_img[v["img"]].append((k, iv))

RED = (0, 0, 255)
GREEN = (0, 200, 0)
made = 0
for img_rel, items in by_img.items():
    items.sort(key=lambda x: x[1])
    full_img = imread_u(DATA / img_rel)
    H, W = full_img.shape[:2]
    dw, dh = 1000, int(H * 1000 / W)
    for k, iv in items[:5]:
        img = cv2.resize(full_img.copy(), (dw, dh))
        gd = base[k]["bbox"]
        loc = (probe[k].get("boxes") or [None])[0]
        # 基线绿框
        p1 = (int(gd[0] * dw), int(gd[1] * dh))
        p2 = (int(gd[2] * dw), int(gd[3] * dh))
        cv2.rectangle(img, p1, p2, GREEN, 2)
        cv2.putText(img, "base", (p1[0], max(16, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
        # Locate 红框
        if loc:
            p1 = (int(loc[0] * dw), int(loc[1] * dh))
            p2 = (int(loc[2] * dw), int(loc[3] * dh))
            cv2.rectangle(img, p1, p2, RED, 4)
            cv2.putText(img, "locate", (p1[0], max(16, p1[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2, cv2.LINE_AA)
        cap = np.full((70, dw, 3), 245, np.uint8)
        cv2.putText(cap, f"{k} | IoU={iv:.2f}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(cap, queries[k]["query"][:95], (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(cap, "red=Locate  green=base", (8, 64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
        out = VIZ / f"{k}.png"
        cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
        made += 1
print(f"生成 {made} 张分歧对比图 -> {VIZ}")