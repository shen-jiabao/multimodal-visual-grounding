# -*- coding: utf-8 -*-
"""列出探针集中的序数/空间问句，为 SoM 探针选定目标。"""
import json
import re

R = r"d:\Desktop\2026AiC"
qw = json.load(open(R + r"\qwen3vl_probe.json", encoding="utf-8"))
merge = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|"
                 r"ninth|tenth|last)\b", re.I)
SPAT = re.compile(r"\b(behind|in front|closest|nearest|leftmost|rightmost|"
                  r"above|below|left|right)\b", re.I)

print("=== 序数问句 ===")
for k, v in qw.items():
    q = v["query"]
    if ORD.search(q):
        print(f"  {k} | {q}")
        print(f"     QW={v.get('bbox')}")
        print(f"     提交={merge[k]['bbox']}  GD={base[k]['bbox']}")

print("\n=== 空间问句(非序数) ===")
for k, v in qw.items():
    q = v["query"]
    if SPAT.search(q) and not ORD.search(q):
        print(f"  {k} | {q[:70]}")