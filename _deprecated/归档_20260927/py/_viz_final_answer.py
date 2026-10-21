# -*- coding: utf-8 -*-
"""为最终提交(guard866)生成可视化抽检图：
红粗=最终提交框；灰细=基线(M)；蓝细=Locate-v2；品红细=GroundingDINO。
挑选：3 条被替换(修大框/风险/典型) + 1 条被护栏拦下 + 3 条随机常规。
输出 viz_out/final_check/*.png
"""
import json
from pathlib import Path

import cv2
import numpy as np

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
VIZ = R / "viz_out" / "final_check"
VIZ.mkdir(parents=True, exist_ok=True)

base = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
final = json.loads((R / "result_gd_3modal_final.json").read_text(encoding="utf-8"))
c885 = json.loads((R / "fusion_conservative.json").read_text(encoding="utf-8"))
l2 = json.loads((R / "predictions_v2_highres_slow.json").read_text(encoding="utf-8"))
g = json.loads((R / "predictions_grounding.json").read_text(encoding="utf-8"))
queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    it = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


changed = [k for k in final if any(abs(a-b) > 1e-6 for a, b in
                                   zip(final[k]["bbox"], base[k]["bbox"]))]
# 885 口径里被替换的集合 - 866 口径里被替换的集合 = 被护栏拦下的 19 条
c885_changed = [k for k in c885 if any(abs(a-b) > 1e-6 for a, b in
                                       zip(c885[k]["bbox"], base[k]["bbox"]))]
rejected = [k for k in c885_changed if k not in changed]
# 替换样本分类
fixes = sorted([k for k in changed if area(base[k]["bbox"]) > 0.35
                and area(final[k]["bbox"]) <= 0.35],
               key=lambda k: area(base[k]["bbox"]), reverse=True)
risky = sorted([k for k in changed if area(final[k]["bbox"]) > 0.35
                and area(base[k]["bbox"]) <= 0.35],
               key=lambda k: area(final[k]["bbox"]), reverse=True)
typical = [k for k in changed if k not in fixes and k not in risky]

keys = sorted(final)
randoms = [keys[i] for i in (1300, 3900, 6900)]
picks = []
if fixes:
    picks.append(("G4修复大框", fixes[0]))
if typical:
    picks.append(("G4典型替换", typical[len(typical)//2]))
picks.append(("G4护栏拦下(无人机,保留基线)", "000247_008"))
picks.append(("基线自身错例(空间query)", "008563_003"))
if rejected:
    picks.append(("G4护栏拦下(其它)", rejected[0]))
for i, k in enumerate(randoms):
    picks.append((f"常规抽检{i+1}", k))

COLORS = {"final": (0, 0, 255), "base": (140, 140, 140),
          "L2": (255, 120, 0), "G": (200, 0, 200)}


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


for tag, qid in picks:
    img = imread_u(DATA / queries[qid]["visible"])
    H, W = img.shape[:2]
    dw, dh = 1000, int(H * 1000 / W)
    img = cv2.resize(img, (dw, dh))
    boxes = [("G", g[qid]["bbox"], 2), ("L2", l2[qid]["bbox"], 2),
             ("base", base[qid]["bbox"], 2), ("final", final[qid]["bbox"], 5)]
    for name, b, th in boxes:
        if name == "base" and iou(base[qid]["bbox"], final[qid]["bbox"]) > 0.999:
            continue
        c = COLORS[name]
        p1 = (int(b[0]*dw), int(b[1]*dh)); p2 = (int(b[2]*dw), int(b[3]*dh))
        cv2.rectangle(img, p1, p2, c, th)
        if name in ("final", "base"):
            cv2.putText(img, name, (max(0, p1[0]), max(18, p1[1]-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    cap = np.full((102, dw, 3), 245, np.uint8)
    cv2.putText(cap, f"[{tag}] {qid}: {queries[qid]['query'][:100]}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    replaced = any(abs(a-b) > 1e-6 for a, b in
                   zip(final[qid]["bbox"], base[qid]["bbox"]))
    cv2.putText(cap, f"G4替换: {'是' if replaced else '否(保留基线)'}   "
                     f"IoU(final,base)={iou(final[qid]['bbox'], base[qid]['bbox']):.3f}",
                (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 200), 1,
                cv2.LINE_AA)
    cv2.putText(cap, f"area  L2={area(l2[qid]['bbox']):.3f}  base(M)={area(base[qid]['bbox']):.3f}  "
                     f"G={area(g[qid]['bbox']):.3f}  final={area(final[qid]['bbox']):.3f}",
                (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 70, 70), 1,
                cv2.LINE_AA)
    cv2.putText(cap, "红粗=final  灰细=base  蓝细=Locate  品红细=GroundingDINO",
                (8, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                cv2.LINE_AA)
    out = VIZ / f"{tag}_{qid}.png"
    cv2.imencode(".png", np.vstack([img, cap]))[1].tofile(str(out))
    print(f"已保存 {out}")
print(f"\n替换总数={len(changed)} 护栏拦下(885-866)={len(rejected)} "
      f"修复大框={len(fixes)} 新增大框={len(risky)}")
