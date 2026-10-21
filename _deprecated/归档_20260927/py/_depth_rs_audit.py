# -*- coding: utf-8 -*-
"""两件事：
1) depth 问句(3041)在最终锁定中的 采用Locate vs 回退基线 分布。
2) RS(380) 结构回退的质量审计：多少是纯结构问句 vs 含结构词的普通问句。
"""
import json
import re
from collections import Counter

R = r"d:\Desktop\2026AiC"
final = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
loc = json.load(open(R + r"\locate_bbox_all.json", encoding="utf-8"))
q = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json",
                   encoding="utf-8"))

DEPTH_NEED = re.compile(
    r"\b(closest|nearest|farthest|furthest|distance|behind|in front|in the "
    r"middle|far |near |faraway|far away|below|above|between|foreground|"
    r"background|depth)\b", re.I)
STRUCTURE_RE = re.compile(
    r"\b(railing|fence|fencing|barrier|guardrail|handrail|railing)\b", re.I)
# 结构词作"主体"的判定：结构词后紧跟 be 动词/现在分词/独立句首，或不与其它对象名词并列
STRUCT_MAIN = re.compile(
    r"\b((a|the|this|that)\s+(metal|wooden|white|black|brown|tall|long|"
    r"concrete|iron|green|gray|light)?\s*(railing|fence|fencing|barrier|"
    r"guardrail|handrail))\b", re.I)

# ---- 1) depth 问句分布 ----
depth_keys = [k for k in q if DEPTH_NEED.search(q[k]["query"])]
cnt = Counter()
for k in depth_keys:
    if final[k]["bbox"] == base[k]["bbox"]:
        cnt["回退基线"] += 1
    else:
        cnt["采用Locate"] += 1
print(f"== depth 问句 {len(depth_keys)} 条在锁定中的分布 ==")
for c, n in cnt.items():
    print(f"  {c}: {n} ({100*n/len(depth_keys):.1f}%)")

# ---- 2) RS 审计 ----
rs_keys = [k for k in q if STRUCTURE_RE.search(q[k]["query"])]
pure = [k for k in rs_keys if STRUCT_MAIN.search(q[k]["query"])]
incidental = [k for k in rs_keys if k not in pure]
print(f"\n== RS 结构词命中 {len(rs_keys)} 条 ==")
print(f"  结构为主语(纯结构问句): {len(pure)}")
print(f"  结构词作背景(疑似误伤): {len(incidental)}")
print(f"\n[纯结构样本]")
for k in pure[:8]:
    print(f"  {k} | {q[k]['query'][:70]}")
print(f"\n[疑似误伤样本]")
for k in incidental[:8]:
    print(f"  {k} | {q[k]['query'][:70]}")