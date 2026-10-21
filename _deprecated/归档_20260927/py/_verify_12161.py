# -*- coding: utf-8 -*-
"""复查 012161_004 已回退基线。"""
import json

R = r"d:\Desktop\2026AiC"
final = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
k = "012161_004"
print(f"changed={final[k]['bbox'] != base[k]['bbox']}")
print(f"融合={final[k]['bbox']}")
print(f"基线={base[k]['bbox']}")