# -*- coding: utf-8 -*-
"""对最终提交 result_gd_3modal_fullseq.json 做可视化抽检：
红粗=最终框(final)；灰细=基线(M v2)。输出 viz_out/spotcheck/{qid}.png
选中：序数覆盖、护栏样本、常规样本。
"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "spotcheck"
VIZ.mkdir(parents=True, exist_ok=True)

final = json.loads((R / "result_gd_3modal_fullseq.json").read_text(encoding="utf-8"))
base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    it = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / ua if ua > 1e-9 else 0.0


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


# 抽检：显式选 7（剩余序数+护栏补漏） + 程序化选 3（修大框/典型替换/随机）
PICKS = [
    # 剩余序数覆盖
    "000023_004", "010019_003", "010033_004", "012826_001", "016774_001",
    # 剩余护栏样本(应零改动)
    "000282_008", "002004_002",
]
# 程序化补充：修大框(基线>0.35→final<=0.35) / 典型替换 / 随机
big_fix = [k for k in final
           if area(base[k]["bbox"]) > 0.35 and area(final[k]["bbox"]) <= 0.35]
replaced = [k for k in final if iou(final[k]["bbox"], base[k]["bbox"]) < 0.999]
if big_fix:
    PICKS.append(sorted(big_fix, key=lambda k: area(base[k]["bbox"]),
                        reverse=True)[0])
if replaced:
    typ = [k for k in replaced if k not in big_fix]
    PICKS.append(sorted(typ, key=lambda k: iou(final[k]["bbox"], base[k]["bbox"]),
                        reverse=True)[len(typ) // 2]
                 if typ else replaced[0])
keys = sorted(final)
PICKS.append(keys[len(keys) // 2])
PICKS = [k for k in dict.fromkeys(PICKS) if k in queries]
RED = (0, 0, 255)
GRAY = (140, 140, 140)

for qid in PICKS:
    if qid not in queries:
        print("[SKIP] 无问句", qid)
        continue
    img = imread_u(DATA / queries[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H * 1000 / W)
    img = cv2.resize(img, (dw, dh))
    fb, bb = final[qid]["bbox"], base[qid]["bbox"]
    # 基线叠画(与final不同才画)
    if iou(fb, bb) < 0.999:
        p1 = (int(bb[0] * dw), int(bb[1] * dh)); p2 = (int(bb[2] * dw), int(bb[3] * dh))
        cv2.rectangle(img, p1, p2, GRAY, 2)
        cv2.putText(img, "base", (p1[0], max(18, p1[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 2, cv2.LINE_AA)
    p1 = (int(fb[0] * dw), int(fb[1] * dh)); p2 = (int(fb[2] * dw), int(fb[3] * dh))
    cv2.rectangle(img, p1, p2, RED, 5)
    cv2.putText(img, "final", (max(0, p1[0]), max(18, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
    cap = np.full((86, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"{qid}: {queries[qid]['query'][:100]}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    changed = iou(fb, bb) < 0.999
    cv2.putText(cap, f"final-area={area(fb):.3f}  base-area={area(bb):.3f}  "
                     f"IoU(final,base)={iou(fb,bb):.3f}  {'<== 已改' if changed else '(同基线)'}",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 200) if changed else (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(cap, "红粗=final  灰细=base", (8, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    out = VIZ / f"{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"已保存 {out}  改={'Y' if changed else 'N'}  "
          f"area final/base={area(fb):.3f}/{area(bb):.3f}")