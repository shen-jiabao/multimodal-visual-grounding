# -*- coding: utf-8 -*-
"""保守替换口径统计：仅当 L2 与 G 互相一致(IoU>=0.5) 且基线 M 与两者都不一致时，
才用 (L2+G)/2 替换 M；其余保持 M 不变。统计覆盖量与影响面。"""
import json
from pathlib import Path

OUT = Path("/root/output")
L2 = json.load(open(OUT / "predictions_v2_highres_slow.json"))
M = json.load(open(OUT / "result_gd_3modal_v2.json"))
G = json.load(open(OUT / "predictions_grounding.json"))
Q = json.load(open("/root/infer/queries.json"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


keys = sorted(set(L2) & set(M) & set(G))
repl, keep = [], 0
big_fixed = 0
for k in keys:
    l2, m, g = L2[k]["bbox"], M[k]["bbox"], G[k]["bbox"]
    if iou(l2, g) >= 0.5 and iou(l2, m) < 0.5 and iou(g, m) < 0.5:
        repl.append(k)
        if area(m) > 0.35:
            big_fixed += 1
    else:
        keep += 1

print(f"总样本 {len(keys)}")
print(f"满足「L2≈G 且 M 孤立」→ 用 (L2+G)/2 替换: {len(repl)} ({100*len(repl)/len(keys):.1f}%)")
print(f"保持基线 M 不变: {keep} ({100*keep/len(keys):.1f}%)")
print(f"其中基线是大框(>0.35)被替换掉的: {big_fixed}")

# 涉及大框统计
big_m = sum(1 for k in keys if area(M[k]["bbox"]) > 0.35)
print(f"基线大框总数: {big_m}")
print("\n替换样本示例(前 12):")
for k in repl[:12]:
    q = Q.get(k, {}).get("query", "")
    print(f"  {k}: M={[round(float(v),3) for v in M[k]['bbox']]}  q={q[:60]}")
