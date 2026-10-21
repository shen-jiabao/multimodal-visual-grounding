# -*- coding: utf-8 -*-
"""SoM vs 提交 对比图（000023 序数五连）。

红粗+编号=Qwen SoM 候选与选中；绿细=当前提交框。
输出 viz_out/som_ordinal/{qid}.png + 一张全候选总览。
"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "som_ordinal"
VIZ.mkdir(parents=True, exist_ok=True)

som = json.loads((R / "qwen3vl_som.json").read_text(encoding="utf-8"))
merge = json.loads((R / "result_locate_merge.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))

img_id = "000023"
img0 = cv2.imdecode(np.fromfile(
    str(DATA / queries[f"{img_id}_001"]["visible"]), dtype=np.uint8),
    cv2.IMREAD_COLOR)
H, W = img0.shape[:2]
dw, dh = 1200, int(H * 1200 / W)
base_img = cv2.resize(img0, (dw, dh))
cands = som[img_id]["candidates"]

RED = (0, 0, 255)
GREEN = (0, 190, 0)
ORANGE = (0, 165, 255)


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    it = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it / ua if ua > 1e-9 else 0.0


def draw_som_candidates(img):
    """画全部候选 + 编号（橙=全部，红=选中由调用方叠加）。"""
    im = img.copy()
    for i, b in enumerate(cands, 1):
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(im, p1, p2, ORANGE, 2)
        cx, cy = (p1[0]+p2[0])//2, (p1[1]+p2[1])//2
        cv2.circle(im, (cx, cy), 16, ORANGE, -1)
        cv2.putText(im, str(i), (cx-8, cy+7), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return im


# 总览图：全部候选编号
overview = draw_som_candidates(base_img)
cv2.putText(overview, "SoM candidates (Qwen3-VL enumerate, L->R)",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ORANGE, 2, cv2.LINE_AA)
cv2.imencode(".png", overview)[1].tofile(str(VIZ / "_overview_candidates.png"))
print("总览 -> _overview_candidates.png")

for qid, r in som[img_id]["queries"].items():
    img = draw_som_candidates(base_img)
    sb = merge[qid]["bbox"]
    # 提交框（绿粗）
    p1 = (int(sb[0]*dw), int(sb[1]*dh)); p2 = (int(sb[2]*dw), int(sb[3]*dh))
    cv2.rectangle(img, p1, p2, GREEN, 4)
    cv2.putText(img, "SUBMIT", (p1[0], max(18, p1[1]-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
    # SoM 选中（红粗）
    ob = r["bbox"]
    if ob:
        p1 = (int(ob[0]*dw), int(ob[1]*dh)); p2 = (int(ob[2]*dw), int(ob[3]*dh))
        cv2.rectangle(img, p1, p2, RED, 4)
        cv2.putText(img, f"SOM#{r['answer_num']}", (max(0, p1[0]), max(18, p1[1]-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2, cv2.LINE_AA)
    cap = np.full((78, dw, 3), 245, np.uint8)
    iv = iou(ob, sb)
    cv2.putText(cap, f"{qid} | SoM选#{r['answer_num']} | "
                     f"IoU(SoM,submit)={iv:.2f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(cap, queries[qid]["query"][:110], (8, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(cap, "orange=SoM candidates(num)  red=SoM chosen  green=current submit",
                (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1,
                cv2.LINE_AA)
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(VIZ / f"{qid}.png"))
print(f"生成 5 张对比图 -> {VIZ}")