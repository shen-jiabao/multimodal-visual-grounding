# -*- coding: utf-8 -*-
"""Qwen3-VL 探针 vs 当前提交 对比可视化。

红粗=Qwen3-VL；绿细=当前提交(result_locate_merge)。
输出 viz_out/qwen3vl_probe/{qid}.png，含问句+IoU。
"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "qwen3vl_probe"
VIZ.mkdir(parents=True, exist_ok=True)

qw = json.loads((R / "qwen3vl_probe.json").read_text(encoding="utf-8"))
merge = json.loads((R / "result_locate_merge.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    it = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


RED = (0, 0, 255)
GREEN = (0, 190, 0)

# 全部 38 条（便于全面看），分歧大的优先
rows = []
for k, v in qw.items():
    if not v.get("bbox"):
        continue
    rows.append((k, iou(v["bbox"], merge[k]["bbox"])))
rows.sort(key=lambda x: x[1])

made = 0
for qid, iv in rows:
    img = imread_u(DATA / queries[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H * 1000 / W)
    img = cv2.resize(img, (dw, dh))
    gb = merge[qid]["bbox"]
    qb = qw[qid]["bbox"]
    # 提交框（绿细）
    p1 = (int(gb[0]*dw), int(gb[1]*dh)); p2 = (int(gb[2]*dw), int(gb[3]*dh))
    cv2.rectangle(img, p1, p2, GREEN, 2)
    cv2.putText(img, "merge", (p1[0], max(16, p1[1]-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)
    # Qwen 框（红粗）
    p1 = (int(qb[0]*dw), int(qb[1]*dh)); p2 = (int(qb[2]*dw), int(qb[3]*dh))
    cv2.rectangle(img, p1, p2, RED, 4)
    cv2.putText(img, "QWEN", (max(0, p1[0]), max(16, p1[1]-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    cap = np.full((72, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"{qid} | IoU(QW,merge)={iv:.2f} | {qw[qid]['time_s']}s",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, queries[qid]["query"][:96], (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(cap, "red=Qwen3-VL   green=current submission",
                (8, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1,
                cv2.LINE_AA)
    out = VIZ / f"{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    made += 1
print(f"生成 {made} 张 -> {VIZ}")
for qid, iv in rows[:13]:
    print(f"  {qid} IoU={iv:.2f}")