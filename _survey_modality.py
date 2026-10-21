# -*- coding: utf-8 -*-
"""query 分类统计：多少条真正需要 depth/IR（距离/远近/热/夜/低光）。

支撑决策：值不值得为 depth/IR 重开管线。
"""
import json
import re
from collections import Counter

R = r"d:\Desktop\2026AiC"
q = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json",
                   encoding="utf-8"))

DEPTH_NEED = re.compile(
    r"\b(closest|nearest|farthest|furthest|distance|behind|in front|in the "
    r"middle|far |near |faraway|far away|below|above|between|foreground|"
    r"background|depth)\b", re.I)
IR_NEED = re.compile(
    r"\b(dark|night|low.?light|dim|shadow|hot|heat|cold|fire|flame|warm|"
    r"glow|lamp|lighted|illuminat|bright)\b", re.I)
SPATIAL = re.compile(
    r"\b(left|right|top|bottom|leftmost|rightmost|topmost|bottommost|upper|"
    r"lower|front|back|north|south|east|west)\b", re.I)

cnt = Counter()
depth_hits = ir_hits = spat_hits = 0
depth_only = ir_only = both = 0
samples_depth = []
samples_ir = []
for k, v in q.items():
    t = v["query"]
    d = bool(DEPTH_NEED.search(t))
    i = bool(IR_NEED.search(t))
    s = bool(SPATIAL.search(t))
    if d:
        depth_hits += 1
        if len(samples_depth) < 6:
            samples_depth.append(k + " | " + t[:70])
    if i:
        ir_hits += 1
        if len(samples_ir) < 6:
            samples_ir.append(k + " | " + t[:70])
    if s:
        spat_hits += 1
    if d and i:
        both += 1
    elif d:
        depth_only += 1
    elif i:
        ir_only += 1

print(f"总查询: {len(q)}")
print(f"\n需要 depth 的问句(距离/远近/前后/上下/之间): {depth_hits} "
      f"({100*depth_hits/len(q):.1f}%)")
print(f"需要 IR 的问句(暗/夜/热/灯/发光): {ir_hits} "
      f"({100*ir_hits/len(q):.1f}%)")
print(f"含空间方位词的问句(left/right/top/...): {spat_hits} "
      f"({100*spat_hits/len(q):.1f}%)")
print(f"\n仅需depth: {depth_only}, 仅需IR: {ir_only}, 两者都要: {both}, "
      f"都不需要: {len(q)-depth_only-ir_only-both}")
print(f"\n[depth 样本]")
for s in samples_depth:
    print("  ", s)
print(f"\n[IR 样本]")
for s in samples_ir:
    print("  ", s)