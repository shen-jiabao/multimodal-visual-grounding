# -*- coding: utf-8 -*-
"""G4 + R2 + R1′ 定稿：
  R1′（优先）：含序数词 且 基线面积>0.2 → 采用 L2 框
  R2：G4 中「集合/复数类 且 缩小替换」→ 回退基线框
输出 /root/output/fusion_g4_r2r1p.json
"""
import json
import re
from collections import Counter

OUT = "/root/output"
L2 = json.load(open(f"{OUT}/predictions_v2_highres_slow.json"))
M = json.load(open(f"{OUT}/result_gd_3modal_v2.json"))
G = json.load(open(f"{OUT}/predictions_grounding.json"))
F = json.load(open(f"{OUT}/fusion_g4.json"))
Q = json.load(open("/root/infer/queries.json"))

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b",
                 re.I)
COLL = re.compile(r"all the|group of|crowd|people|men|women|children|students|tourists|"
                  r"\bpair\b|\btwo\b|\bthree\b|\bfour\b|\bfive\b|\bboth\b", re.I)


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = area(a)+area(b)-it
    return it/ua if ua > 1e-9 else 0.


def norm(b):
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0., min(1., x1)), max(0., min(1., y1))
    x2, y2 = max(0., min(1., x2)), max(0., min(1., y2))
    return [x1, y1, min(1., max(x2, x1 + 1e-3)), min(1., max(y2, y1 + 1e-3))]


out, applied = {}, Counter()
for k, rec in F.items():
    b = rec["bbox"]
    rule = None
    # R1′ 优先：序数词 且 基线面积 > 0.2
    if k in L2 and ORD.search(Q[k]["query"]) and area(M[k]["bbox"]) > 0.2:
        b, rule = norm(L2[k]["bbox"]), "R1p_ordinal"
    # R2：G4 替换中，集合/复数类且缩小 → 回退基线
    elif (iou(F[k]["bbox"], M[k]["bbox"]) < 0.999
          and area(F[k]["bbox"]) < area(M[k]["bbox"])
          and COLL.search(Q[k]["query"])):
        b, rule = M[k]["bbox"], "R2_collective"
    entry = dict(rec)
    entry["bbox"] = [round(float(v), 6) for v in b]
    out[k] = entry
    if rule:
        applied[rule] += 1

p = f"{OUT}/fusion_g4_r2r1p.json"
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

chg = [k for k in out if iou(out[k]["bbox"], F[k]["bbox"]) < 0.999]
bigF = sum(1 for v in F.values() if area(v["bbox"]) > 0.35)
bigN = sum(1 for v in out.values() if area(v["bbox"]) > 0.35)
CONF = ["000282_004", "000282_008", "002004_001", "002004_002"]
print("=== G4 + R2 + R1′ ===")
for r, c in applied.most_common():
    print(f"  {r:16s} {c}")
print(f"相对 G4 改变 {len(chg)} 条")
print(f"大框: G4 {bigF} → 新 {bigN}")
print(f"护栏样本是否被改: {[(k, k in chg) for k in CONF]}")
for k in ["000023_002", "000023_003", "000023_004", "000023_006", "000478_005"]:
    print(f"  {k}: M={area(M[k]['bbox']):.3f} G4={area(F[k]['bbox']):.3f} "
          f"→ 新={area(out[k]['bbox']):.3f}")
print(f"输出: {p}")
