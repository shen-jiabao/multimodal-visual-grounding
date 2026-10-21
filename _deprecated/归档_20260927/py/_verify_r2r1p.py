# -*- coding: utf-8 -*-
"""本地复核 G4+R2+R1′ 定稿，并出 5 张抽检图。"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
new = json.loads((R / "fusion_g4_r2r1p.json").read_text(encoding="utf-8"))
g4 = json.loads((R / "fusion_g4.json").read_text(encoding="utf-8"))
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
Q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
test = json.loads((DATA / "test.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = area(a)+area(b)-it
    return it/ua if ua > 1e-9 else 0.


CONF = ["000282_004", "000282_008", "002004_001", "002004_002"]
chg = [k for k in new if iou(new[k]["bbox"], g4[k]["bbox"]) < 0.999]
print(f"条目 {len(new)}; key 缺失 {len(set(test)-set(new))} 多余 {len(set(new)-set(test))}")
bad = [k for k, v in new.items()
       if not (0 <= v["bbox"][0] < v["bbox"][2] <= 1
               and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
print(f"非法 bbox {len(bad)}; 相对 G4 改变 {len(chg)} (期望 96)")
print(f"大框 {sum(1 for v in new.values() if area(v['bbox'])>0.35)} (G4 348)")
for k in CONF:
    print(f"  {k}: 改动={'是' if k in chg else '否'} "
          f"IoU(new,base)={iou(new[k]['bbox'], base[k]['bbox']):.3f}")

VIZ = R / "viz_out" / "review_v1"
VIZ.mkdir(parents=True, exist_ok=True)
CASES = [("序数-2nd pier", "000023_002"), ("序数-3rd pier", "000023_003"),
         ("序数-4th pier", "000023_004"), ("空间-对照(未改)", "000023_006"),
         ("集合-回退大框", "000478_005")]
COLORS = {"new": (0, 0, 255), "g4": (0, 200, 200), "M": (150, 150, 150),
          "L2": (255, 120, 0)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


for tag, qid in CASES:
    img = imread_u(DATA / Q[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H*1000/W)
    img = cv2.resize(img, (dw, dh))
    for name, b, th, c in [("L2", l2[qid]["bbox"], 2, COLORS["L2"]),
                           ("M", base[qid]["bbox"], 2, COLORS["M"]),
                           ("g4", g4[qid]["bbox"], 2, COLORS["g4"]),
                           ("new", new[qid]["bbox"], 5, COLORS["new"])]:
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, th)
        if name in ("new", "g4"):
            cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cap = np.full((102, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[{tag}] {qid}: {Q[qid]['query'][:100]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, f"IoU(new,G4)={iou(new[qid]['bbox'], g4[qid]['bbox']):.3f}   "
                     f"area  L2={area(l2[qid]['bbox']):.3f}  M={area(base[qid]['bbox']):.3f}  "
                     f"G4={area(g4[qid]['bbox']):.3f}  new={area(new[qid]['bbox']):.3f}",
                (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(cap, "红粗=new  黄细=G4  灰细=M(base,非真值)  蓝细=Locate",
                (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"{tag}_{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"已保存 {out}")
