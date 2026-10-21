# -*- coding: utf-8 -*-
"""校验 R1″ 新版并出前后对比图。"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
new = json.loads((R / "fusion_g4_r2r1p_r1pp.json").read_text(encoding="utf-8"))
cur = json.loads((R / "fusion_g4_r2r1p.json").read_text(encoding="utf-8"))
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
Q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
test = json.loads((DATA / "test.json").read_text(encoding="utf-8"))
FIELDS = {"visible", "infrared", "depth", "query", "bbox"}


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


print(f"条目 {len(new)}; key 缺失 {len(set(test)-set(new))} 多余 {len(set(new)-set(test))}")
print(f"字段异常 {sum(1 for v in new.values() if set(v) != FIELDS)}; "
      f"非法 bbox {sum(1 for v in new.values() if not (0 <= v['bbox'][0] < v['bbox'][2] <= 1 and 0 <= v['bbox'][1] < v['bbox'][3] <= 1))}")

VIZ = R / "viz_out" / "review_v3"
VIZ.mkdir(parents=True, exist_ok=True)
CASES = ["000023_006", "001822_004", "001647_002", "003201_002", "001941_004"]
COLORS = {"new": (0, 0, 255), "L2": (255, 120, 0), "M": (150, 150, 150),
          "cur": (0, 200, 200)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


for qid in CASES:
    img = imread_u(DATA / Q[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H*1000/W)
    img = cv2.resize(img, (dw, dh))
    for name, b, th in [("M", base[qid]["bbox"], 2), ("cur", cur[qid]["bbox"], 2),
                        ("L2", l2[qid]["bbox"], 2), ("new", new[qid]["bbox"], 5)]:
        c = COLORS[name]
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, th)
        if name in ("new", "cur"):
            cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cap = np.full((102, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[R1″ 已采纳] {qid}: {Q[qid]['query'][:100]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"area  M(base)={area(base[qid]['bbox']):.3f}  锁定版={area(cur[qid]['bbox']):.3f}  "
                     f"L2={area(l2[qid]['bbox']):.3f}  新版={area(new[qid]['bbox']):.3f}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(cap, "红粗=新版  黄细=锁定版  灰细=M(base)  蓝细=Locate",
                (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"R1pp_applied_{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"已保存 {out}")
