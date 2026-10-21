# -*- coding: utf-8 -*-
"""改进版核心词抽取：取序数词之后的头名词短语（去掉空间/关系尾巴）。

枚举只需类别，空间限定交给 SoM 选择环节。
统计：改进后 仍需 VLM 兜底 的条数（空/极短/仍过长）。
"""
import json
import os
import re
from collections import Counter, defaultdict

R = r"d:\Desktop\2026AiC"
Q = json.load(open(os.path.join(
    R, "初赛数据集-基于大模型的多模态视觉理解与推理", "queries",
    "queries.json"), encoding="utf-8"))

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|"
                 r"ninth|tenth|last)\b", re.I)

# 头名词短语的终止标记（出现即截断）
CUT = re.compile(
    r"\s*[,;]\s*|\b(located|which|that|who|standing|sitting|walking|seen|"
    r"positioned|visible|in front of|behind|next to|beside|near|"
    r"on the|at the|to the|inside|outside|under|above|below|"
    r"with|holding|wearing|facing|leaning|riding|carrying|parked|"
    r"closest|farthest|nearest|furthest|from|between)\b", re.I)
ART = re.compile(r"^\s*(the|a|an|its|his|her|their|this|that)\s+", re.I)
MAXW = 5


def extract_head(q: str) -> str:
    """取序数词之后的头名词短语（≤5 词）。"""
    q = re.sub(r"\([^)]*\)", " ", q)          # 去括号内容
    m = ORD.search(q)
    seg = q[m.end():] if m else q             # 序数词之后的片段
    seg = ART.sub("", seg).strip()
    seg = CUT.split(seg)[0].strip()           # 截断到终止标记
    seg = re.sub(r"[^\w\s\-']", " ", seg)
    seg = re.sub(r"\s+", " ", seg).strip()
    words = seg.split()
    return " ".join(words[:MAXW])


cores, need_vlm = {}, []
for k, v in Q.items():
    if not ORD.search(v["query"]):
        continue
    c = extract_head(v["query"])
    cores[k] = c
    if len(c) < 3 or len(c.split()) > MAXW:
        need_vlm.append(k)

ordinal = [k for k in cores]
print(f"序数问句 {len(ordinal)}")
print(f"改进后需 VLM 兜底(空/超长): {len(need_vlm)} "
      f"({100*len(need_vlm)/len(ordinal):.1f}%)")
print("\n=== 改进后样例（前 28）===")
for k in ordinal[:28]:
    print(f"  [{cores[k]}]  <-  {Q[k]['query'][:76]}")
print("\n=== 需兜底样例 ===")
for k in need_vlm[:10]:
    print(f"  [{cores[k]}]  <-  {Q[k]['query'][:76]}")

# 分组（改进后）
groups = defaultdict(set)
for k, c in cores.items():
    groups[(os.path.basename(Q[k]["visible"]), c.lower())].add(k)
print(f"\n改进后唯一 (图, 核心词) 组: {len(groups)}")
print(f"SoM 总调用 = {len(groups)} + {len(ordinal)} = "
      f"{len(groups)+len(ordinal)}；≈{(len(groups)+len(ordinal))*3/3600:.1f}h")

json.dump(cores, open(os.path.join(R, "_ordinal_cores_v2.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)
print("已存 _ordinal_cores_v2.json")