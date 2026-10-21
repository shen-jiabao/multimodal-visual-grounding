# -*- coding: utf-8 -*-
"""序数问句分析 + 核心名词启发式抽取。

目的：估算 SoM 全量的调用量（按 (图, 核心词) 分组的枚举次数），并评估启发式抽取质量。
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

# 需剥离的序数/空间/指令性片段
STRIP_PATTERNS = [
    r"\bfrom left to right\b", r"\bfrom right to left\b",
    r"\bfrom top to bottom\b", r"\bfrom bottom to top\b",
    r"\bfrom the left\b", r"\bfrom the right\b",
    r"\bleft to right\b", r"\bright to left\b",
    r"\b(top to bottom|bottom to top)\b",
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b",
    r"^\s*(count|number|outline|identify|find|locate|select)\s+(the\s+)?",
    r"\bcount\s+(the\s+)?",
    r"^\s*(the|a|an)\s+",
    r"\bthat is\b.*$", r"\bwhich is\b.*$", r"\blocated\b.*$",
    r"\bpositioned\b.*$", r"\bstanding\b.*$", r"\bsitting\b.*$",
    r"\bseen from\b.*$",
]
LEAD_TRAIL = re.compile(r"^[\s,.;:\-]+|[\s,.;:\-]+$")


def extract_core(q: str) -> str:
    """启发式抽取核心名词短语。"""
    t = q
    for p in STRIP_PATTERNS:
        t = re.sub(p, " ", t, flags=re.I)
    t = LEAD_TRAIL.sub("", t)
    t = re.sub(r"\s+", " ", t)
    # 去掉残留的关系从句（含 of/with/on/at/in 的长尾）
    return t.strip()


ordinal = {k: v["query"] for k, v in Q.items() if ORD.search(v["query"])}
print(f"总问句 {len(Q)}")
print(f"序数问句 {len(ordinal)} ({100*len(ordinal)/len(Q):.1f}%)\n")

groups = defaultdict(set)
cores = {}
for k, q in ordinal.items():
    img = os.path.basename(Q[k]["visible"])
    c = extract_core(q)
    cores[k] = c
    groups[(img, c.lower())].add(k)

print(f"唯一 (图, 核心词) 分组: {len(groups)}")
print(f"→ SoM 枚举调用次数 = {len(groups)}（同组共享一次枚举）")
print(f"→ SoM 选择调用次数 = {len(ordinal)}（每问句一次）")
tot = len(groups) + len(ordinal)
print(f"→ SoM 总调用 {tot}；按 ~3s/次 ≈ {tot*3/3600:.1f}h")
print(f"→ 加上全量 9555 直接推理 {9555*2.46/3600:.1f}h\n")

# 分组分布
sizes = Counter(len(v) for v in groups.values())
print("每组问句数分布:", dict(sorted(sizes.items())))
print("\n=== 核心词抽取样例（前 30，供检查）===")
for k in list(ordinal)[:30]:
    print(f"  [{cores[k]}]  <-  {ordinal[k][:78]}")

# 可疑：抽取为空或过长
empty = [k for k, c in cores.items() if not c or len(c) < 3]
long = [k for k, c in cores.items() if len(c) > 40]
print(f"\n可疑(空/过短): {len(empty)}  可疑(过长>40): {len(long)}")
for k in (empty + long)[:10]:
    print(f"  [{cores[k]}]  <-  {ordinal[k][:78]}")

json.dump(cores, open(os.path.join(R, "_ordinal_cores_heur.json"), "w",
                      encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n已存 _ordinal_cores_heur.json（{len(cores)} 条）")