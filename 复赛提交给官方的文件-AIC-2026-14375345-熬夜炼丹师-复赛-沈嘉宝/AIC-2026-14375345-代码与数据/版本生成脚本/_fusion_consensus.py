# -*- coding: utf-8 -*-
"""A 方案：家族共识/多数投票（纯 CPU）。
与用户给的脚本逻辑一致，仅修正「取 bbox 字段」的 bug，并追加差异分桶统计。
"""
import json
import re
from collections import Counter

OUT = "/root/output"
L2 = json.load(open(f"{OUT}/predictions_v2_highres_slow.json"))
M = json.load(open(f"{OUT}/result_gd_3modal_v2.json"))      # 现提交基线(软链)
G = json.load(open(f"{OUT}/predictions_grounding.json"))
Q = json.load(open("/root/infer/queries.json"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.0


def avg(a, b, wa=0.5):
    return [wa*a[i] + (1-wa)*b[i] for i in range(4)]


HARD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b|"
                  r"\b(two|three|four|five|six|seven|eight|nine|ten|pair|both)\b|"
                  r"from (left|right) to (left|right)", re.I)


def is_hard(q):
    return bool(HARD.search(q or ""))


def norm(b):
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    return [x1, y1, x2, y2]


keys = sorted(set(L2) & set(M) & set(G))
fusion, rules, diffs = {}, Counter(), []
rule_by_bucket = Counter()
for k in keys:
    l2, m, g = L2[k]["bbox"], M[k]["bbox"], G[k]["bbox"]
    i_lm, i_lg, i_mg = iou(l2, m), iou(l2, g), iou(m, g)
    q = (Q.get(k, {}) or {}).get("query", "")
    if i_mg >= 0.5:
        base = avg(m, g, 0.5)
        if iou(l2, base) >= 0.5:
            box, r = avg(l2, base, 0.6), "LMG_consensus"
        else:
            box, r = base, "MG_consensus"
    elif max(i_lm, i_lg) >= 0.5:
        gd = m if i_lm >= i_lg else g
        box, r = avg(l2, gd, 0.6), "L2_GD_pair"
    else:
        if is_hard(q):
            box, r = m, "hard_route_M"
        else:
            box, r = l2, "hard_route_L2"
    fusion[k] = norm(box)
    rules[r] += 1
    iv = iou(fusion[k], m)
    if iv < 0.99:
        diffs.append((k, round(iv, 3), r, q[:60]))
    b = ("完全相同>=0.99" if iv >= 0.99 else
         "轻微0.5~0.99" if iv >= 0.5 else
         "中等0.1~0.5" if iv >= 0.1 else "显著<0.1")
    rule_by_bucket[(r, b)] += 1

json.dump(fusion, open(f"{OUT}/fusion_consensus.json", "w"), indent=1)
print("=== 规则分布 ===")
for r, c in rules.most_common():
    print(f"  {r:20s} {c} ({100*c/len(keys):.1f}%)")
print(f"\n总样本 {len(keys)}，与现提交(M) 有差异(<0.99)的样本 {len(diffs)}")

print("\n=== 相对现提交的差异分桶 ===")
buckets = Counter()
for r, b in rule_by_bucket:
    buckets[b] += rule_by_bucket[(r, b)]
for b, c in buckets.most_common():
    print(f"  {b:14s} {c} ({100*c/len(keys):.1f}%)")

print("\n=== 规则 × 差异桶 ===")
for (r, b), c in sorted(rule_by_bucket.items(), key=lambda x: (-x[1])):
    print(f"  {r:20s} {b:14s} {c}")

if diffs:
    print("\n=== 差异样本(按 IoU 升序前 20) ===")
    for k, iv, r, q in sorted(diffs, key=lambda t: t[1])[:20]:
        print(f"  {k}  IoU(fus,M)={iv}  [{r}]  {q}")
