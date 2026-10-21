# -*- coding: utf-8 -*-
"""抽检可视化：从 234 条分歧样本里挑 6 条（CLIP 分别选中 M/L2/G 各 2 条），
把三个候选框 + 选中框画在可见光图上，输出到 /root/output/pilot_viz/。"""
import json
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理")
OUT = Path("/root/output")
VIZ = OUT / "pilot_viz"

pilot = json.loads((OUT / "fusion_pilot_235.json").read_text(encoding="utf-8"))
detail = json.loads((OUT / "fusion_detail.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
VIZ.mkdir(exist_ok=True)


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


# 按选中来源分组，每组挑「CLIP 前三名分差最小（最难判）」的 2 条
groups = {"L2": [], "M": [], "G": []}
for qid, r in pilot.items():
    src = r.get("source", "")
    if not src.startswith("clip:"):
        continue
    key = src.split(":", 1)[1]
    if key not in groups or "sim" not in r:
        continue
    sims = sorted(r["sim"].values(), reverse=True)
    margin = sims[0] - sims[1] if len(sims) > 1 else 1.0
    bx = detail[qid]["boxes"]
    spread = 1 - max(iou(bx["L2_locate_v2_hires"], bx["M_gd3modal"]),
                     iou(bx["L2_locate_v2_hires"], bx["G_groundingdino"]),
                     iou(bx["M_gd3modal"], bx["G_groundingdino"]))
    groups[key].append((margin, -spread, qid))

picks = []
for key in ("L2", "M", "G"):
    groups[key].sort()
    picks.extend([q for _, _, q in groups[key][:2]])
print("抽检样本:", picks)

COLORS = {"L2": (255, 120, 0), "M": (0, 200, 0), "G": (200, 0, 200),
          "L1": (150, 150, 150)}
KEY2TAG = {"L1_locate_v1": "L1", "L2_locate_v2_hires": "L2",
           "M_gd3modal": "M", "G_groundingdino": "G"}

for qid in picks:
    r = pilot[qid]
    rec = queries[qid]
    img = imread_u(DATA / rec["visible"])
    H, W = img.shape[:2]
    dw = 1000
    dh = int(H * dw / W)
    img = cv2.resize(img, (dw, dh))

    # 三个候选框
    for key, b in detail[qid]["boxes"].items():
        tag = KEY2TAG.get(key)
        if tag is None:
            continue
        c = COLORS[tag]
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, 2)
        sim = r.get("sim", {}).get(tag)
        lab = f"{tag}" + (f" sim={sim:.3f}" if sim is not None else "")
        cv2.putText(img, lab, (max(0, p1[0]), max(18, p1[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    # 选中框（黄粗线）
    b = r["bbox"]
    p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
    cv2.rectangle(img, p1, p2, (0, 255, 255), 5)
    cv2.putText(img, f"PICK={r['source']}", (max(0, p1[0]), max(44, p1[1]+22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    cap = np.full((72, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"{qid}: {rec['query'][:108]}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"chosen={r['source']}  sims={r.get('sim')}", (8, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    panel = np.vstack([img, cap])
    out = VIZ / f"{qid}_fusion.png"
    cv2.imencode(".png", panel)[1].tofile(str(out))
    print("已保存", out)
