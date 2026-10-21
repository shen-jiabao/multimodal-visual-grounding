# -*- coding: utf-8 -*-
"""对比抽样：把远端结果与本地结果的两个框画在同一张可见光图上。
红=远端(全量结果)，绿=本地(partial)，并在下方附 query 文字。
"""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"d:\Desktop\2026AiC")
DATA = ROOT / "初赛数据集-基于大模型的多模态视觉理解与推理"
OUT = ROOT / "viz_out"
QIDS = ["000261_006", "001353_001", "000912_003", "000513_004"]

remote = json.loads((ROOT / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
local = json.loads((ROOT / "result_gd_3modal_local_partial.json").read_text(encoding="utf-8"))


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


OUT.mkdir(exist_ok=True)
for qid in QIDS:
    if qid not in remote or qid not in local:
        print(f"[跳过] {qid} 不在两份结果中")
        continue
    rec = remote[qid]
    img = imread_u(DATA / rec["visible"])
    H, W = img.shape[:2]
    dw = 900
    dh = int(H * dw / W)
    img = cv2.resize(img, (dw, dh))

    for box, color, name in [(rec["bbox"], (0, 0, 255), "REMOTE"),
                             (local[qid]["bbox"], (0, 200, 0), "LOCAL")]:
        x1, y1, x2, y2 = [float(v) for v in box]
        p1 = (int(x1 * dw), int(y1 * dh))
        p2 = (int(x2 * dw), int(y2 * dh))
        cv2.rectangle(img, p1, p2, color, 3)
        cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    cap = np.full((64, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"{qid}: {rec['query'][:110]}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"remote={[round(float(v),3) for v in rec['bbox']]}  "
                     f"local={[round(float(v),3) for v in local[qid]['bbox']]}",
                (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1,
                cv2.LINE_AA)
    panel = np.vstack([img, cap])
    out = OUT / f"{qid}_cmp.png"
    cv2.imencode(".png", panel)[1].tofile(str(out))
    print(f"已保存 {out}")

    # 放大裁剪：两框并集 + 20% 边距
    rb, lb = [float(v) for v in rec["bbox"]], [float(v) for v in local[qid]["bbox"]]
    ux1, uy1 = min(rb[0], lb[0]), min(rb[1], lb[1])
    ux2, uy2 = max(rb[2], lb[2]), max(rb[3], lb[3])
    pw, ph = (ux2 - ux1) * 0.25, (uy2 - uy1) * 0.25
    cx1 = max(0.0, ux1 - pw); cy1 = max(0.0, uy1 - ph)
    cx2 = min(1.0, ux2 + pw); cy2 = min(1.0, uy2 + ph)
    full = imread_u(DATA / rec["visible"])
    Hf, Wf = full.shape[:2]
    crop = full[int(cy1 * Hf):int(cy2 * Hf), int(cx1 * Wf):int(cx2 * Wf)]
    if crop.size:
        sc = max(1.0, 700.0 / max(crop.shape[1], 1))
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
        ch, cw = crop.shape[:2]
        for box, color, name in [(rb, (0, 0, 255), "REMOTE"),
                                 (lb, (0, 200, 0), "LOCAL")]:
            p1 = (int((box[0] - cx1) / (cx2 - cx1) * cw),
                  int((box[1] - cy1) / (cy2 - cy1) * ch))
            p2 = (int((box[2] - cx1) / (cx2 - cx1) * cw),
                  int((box[3] - cy1) / (cy2 - cy1) * ch))
            cv2.rectangle(crop, p1, p2, color, 3)
            cv2.putText(crop, name, (max(0, p1[0]), max(20, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        zout = OUT / f"{qid}_zoom.png"
        cv2.imencode(".png", crop)[1].tofile(str(zout))
        print(f"已保存 {zout}")
