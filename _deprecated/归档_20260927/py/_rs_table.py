# -*- coding: utf-8 -*-
"""RS 疑似误伤 165 条判读表：
query 原文 | 结构词 | 结构词上下文(介词框架) | 头名词 | Locate框 | 基线框 | 面积比
"""
import json
import re

R = r"d:\Desktop\2026AiC"
final = json.load(open(R + r"\result_locate_merge.json", encoding="utf-8"))
base = json.load(open(R + r"\result_gd_3modal_fullseq.json", encoding="utf-8"))
loc = json.load(open(R + r"\locate_bbox_all.json", encoding="utf-8"))
q = json.load(open(R + r"\初赛数据集-基于大模型的多模态视觉理解与推理\queries\queries.json",
                   encoding="utf-8"))

STRUCTURE_RE = re.compile(
    r"\b(railing|fence|fencing|barrier|guardrail|handrail|railing)\b", re.I)
STRUCT_MAIN = re.compile(
    r"\b((a|the|this|that)\s+(metal|wooden|white|black|brown|tall|long|"
    r"concrete|iron|green|gray|light)?\s*(railing|fence|fencing|barrier|"
    r"guardrail|handrail))\b", re.I)
PREP = re.compile(
    r"\b(in front of|behind|next to|near |leaning on|on|against|along|"
    r"beside|by|under|above|between|attached to|mounted on|around|"
    r"separating|visible between|at the (base|top))\b", re.I)

# 头名词近似：query 去掉修饰后最后一个名词。用简单启发——分词后取最后两个实词
STOP = set("a an the of in on at by to from with and or for this that these "
           "those is are was were has have had be being standing sitting "
           "walking running wearing located positioned visible painted "
           "covered resting mounted attached leaning separated".split())


def head_noun(s):
    """极简头名词近似：去掉停用词后最后两个实词。"""
    words = [w.lower().strip(".,'") for w in s.split()
             if w.lower().strip(".,'") not in STOP]
    return " ".join(words[-2:]) if words else s


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


rows = []
for k in q:
    t = q[k]["query"]
    if not STRUCTURE_RE.search(t):
        continue
    if STRUCT_MAIN.search(t):
        continue  # 纯结构主语，不在误伤候选
    # 结构词 + 上下文
    m = STRUCTURE_RE.search(t)
    word = m.group(1)
    pm = PREP.search(t)
    rows.append({
        "qid": k,
        "query": t,
        "word": word,
        "prep": pm.group(0) if pm else "",
        "head": head_noun(t),
        "loc_bbox": loc.get(k, {}).get("bbox"),
        "base_bbox": base[k]["bbox"],
        "area_ratio": (round(area(loc[k]["bbox"]) / area(base[k]["bbox"]), 2)
                       if loc.get(k, {}).get("bbox") else None),
    })

print(f"疑似误伤候选 {len(rows)} 条\n")
print("=" * 120)
for r in rows:
    loc_b = r["loc_bbox"]
    lb = [round(v, 3) for v in loc_b] if loc_b else "NOBOX"
    bb = [round(v, 3) for v in r["base_bbox"]]
    print(f"[{r['qid']}] 头名词≈[{r['head']}] 结构词={r['word']} "
          f"prep={r['prep']!r} area比={r['area_ratio']}")
    print(f"    Q: {r['query'][:95]}")
    print(f"    Locate={lb}  Base={bb}")

json.dump(rows, open(r"d:\Desktop\2026AiC\_rs_audit_table.json", "w",
                     encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n已存 _rs_audit_table.json")