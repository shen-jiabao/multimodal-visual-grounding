#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 shrink 件：每条边是否真的内缩（x1/y1 变大、x2/y2 变小）。"""
import json
from pathlib import Path

SUB = Path(r"d:\Desktop\2026AiC\submission_zips\semifinal\未提交")
base = json.loads((SUB / "result_semi_pick_v1.json").read_text("utf-8"))
shr = json.loads((SUB / "result_semi_27b_shrink.json").read_text("utf-8"))

n = len(shr)
edge_in = [0, 0, 0, 0]      # x1增 / y1增 / x2减 / y2减
area_smaller = 0
area_same = 0
area_bigger = 0
all4 = 0
none = 0
for k, e in shr.items():
    b = base[k]["bbox"]
    a, c = e["bbox"], b
    inc = [a[0] > c[0], a[1] > c[1], a[2] < c[2], a[3] < c[3]]
    for i, v in enumerate(inc):
        if v:
            edge_in[i] += 1
    if all(inc):
        all4 += 1
    if not any(inc):
        none += 1
    ar_b = (c[2] - c[0]) * (c[3] - c[1])
    ar_a = (a[2] - a[0]) * (a[3] - a[1])
    if ar_a < ar_b:
        area_smaller += 1
    elif ar_a == ar_b:
        area_same += 1
    else:
        area_bigger += 1

print(f"总条数 {n}")
print(f"x1 内缩 {edge_in[0]}｜y1 内缩 {edge_in[1]}｜"
      f"x2 内缩 {edge_in[2]}｜y2 内缩 {edge_in[3]}")
print(f"四边全部内缩 {all4}｜无任何边内缩 {none}")
print(f"面积变小 {area_smaller}｜不变 {area_same}｜变大 {area_bigger}")

# 列出未四边内缩的条目
odd = []
for k, e in shr.items():
    b = base[k]["bbox"]
    a, c = e["bbox"], b
    if not (a[0] > c[0] and a[1] > c[1] and a[2] < c[2] and a[3] < c[3]):
        odd.append((k, c, a))
print(f"\n未完全四边内缩的条目 {len(odd)} 个（前 10）：")
for k, c, a in odd[:10]:
    w_b, w_a = round(c[2]-c[0], 4), round(a[2]-a[0], 4)
    h_b, h_a = round(c[3]-c[1], 4), round(a[3]-a[1], 4)
    print(f"  {k}: 基体{c} w×h={w_b}×{h_b} -> shrink{a} w×h={w_a}×{h_a}")
