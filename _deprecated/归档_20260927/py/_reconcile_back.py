# -*- coding: utf-8 -*-
"""回退对账：对每条回退基线的 qid 判定实际规则，与 back_to_base=3001 对账。

定位 README 数字差：规则加总 2709 vs 实际回退 3001，差 292 条。
"""
import json
import re

R = r"d:\Desktop\2026AiC"
loc = json.load(open(R + r"\locate_bbox_all.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
q = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json",
                   encoding="utf-8"))

GUARD = {"000282_004", "000282_008", "002004_001", "002004_002"}
ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|"
                 r"eighth|ninth|tenth|last)\b", re.I)
REGION_RE = re.compile(
    r"\b(lake|water|pond|river|sea|ocean|sky|cloud|grass|lawn|ground|floor|"
    r"road|street|sidewalk|path|field|sand|snow|wall|fence|barrier|railing|"
    r"bamboo|tree|door|area|region|scenery|background|foreground|landscape)\b",
    re.I)
STRUCTURE_RE = re.compile(
    r"\b(railing|fence|fencing|barrier|guardrail|handrail|railing)\b", re.I)
NEAR_RE = re.compile(r"\b(closest|nearest|near (the|a)|in the (front|foreground))\b",
                     re.I)
SPAT = re.compile(r"\b(behind|in front|next to|near |below|above|beneath|"
                  r"under |closest|farthest|furthest|between|left of|right of|"
                  r"front|back|foreground|background|upper|lower|far |near)\b", re.I)


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    it = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - it
    return it / ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


counts = {}
from collections import Counter
cnt = Counter()
for k, rec in base.items():
    lb = loc.get(k, {}).get("bbox")
    bb = rec["bbox"]
    text = q[k]["query"]
    # 与 _merge_locate_gd.py 完全同序判定
    if k in GUARD:
        cnt["GUARD"] += 1
    elif STRUCTURE_RE.search(text):
        cnt["RS_structure"] += 1
    elif ORD.search(text):
        cnt["R3_ordinal"] += 1
    elif lb is None:
        cnt["R0_noframe"] += 1
    elif area(lb) > 0.35 and not REGION_RE.search(text):
        cnt["R1_bigarea"] += 1
    elif iou(lb, bb) < 0.2 and not REGION_RE.search(text):
        if area(lb) > 1.5 * area(bb) and not NEAR_RE.search(text) \
                and not (ORD.search(text) or SPAT.search(text)):
            cnt["R2_drift"] += 1
        else:
            cnt["KB_keep_locate"] += 1   # 豁免后仍采用 Locate
    else:
        cnt["KEEP_LOCATE"] += 1

print("== 全部 9555 判定分布 ==")
for r, c in cnt.most_common():
    print(f"  {r:16s} {c}")
total_back = cnt["GUARD"] + cnt["RS_structure"] + cnt["R3_ordinal"] \
    + cnt["R0_noframe"] + cnt["R1_bigarea"] + cnt["R2_drift"]
print(f"\n回退合计: {total_back}")
print(f"采用 Locate 合计: {cnt['KEEP_LOCATE'] + cnt['KB_keep_locate']}")
print(f"总计: {total_back + cnt['KEEP_LOCATE'] + cnt['KB_keep_locate']}")
print(f"\n与锁定版比对: 回退应=3001, 采用应=6554")