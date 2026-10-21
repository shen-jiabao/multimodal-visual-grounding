# -*- coding: utf-8 -*-
"""验证序数类已回退基线、SPAT/普通类仍采用 Locate。"""
import json
import re

R = r"d:\Desktop\2026AiC"
final = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
q = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json",
                   encoding="utf-8"))
ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|"
                 r"eighth|ninth|tenth|last|leftmost|rightmost|topmost|"
                 r"bottommost)\b", re.I)

print("== 关键序数样本（应回退基线） ==")
for k in ["000023_002", "000023_003", "000023_004", "016774_001"]:
    same = final[k]["bbox"] == base[k]["bbox"]
    print(f"  {k} [{q[k]['query'][:50]}] 回退基线={same}")
print("== 用户判定 Locate 更对的样本（应保留 Locate） ==")
for k in ["001038_003", "000289_001", "014387_003"]:
    same = final[k]["bbox"] == base[k]["bbox"]
    print(f"  {k} [{q[k]['query'][:50]}] 保留Locate={not same}")
# 序数类一致率统计
allord = [k for k, v in q.items() if ORD.search(v["query"])]
keep = sum(1 for k in allord if final[k]["bbox"] == base[k]["bbox"])
print(f"\n序数类总数 {len(allord)}，回退基线 {keep} "
      f"({100*keep/len(allord):.1f}%)")