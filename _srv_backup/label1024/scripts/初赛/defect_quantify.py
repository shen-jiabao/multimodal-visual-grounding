import json
import re
from collections import Counter

BASE = r"E:\AIC人工智能算法精英大赛-基于大模型的多模态视觉理解与推理赛道"

with open(BASE + r"\dataset\初赛\queries\queries.json", encoding="utf-8") as f:
    data = json.load(f)

queries = {qid: v["query"] for qid, v in data.items()}
total = len(queries)
print("总查询:", total)

SPATIAL = [
    "leftmost", "rightmost", "behind", "beside", "above", "below", "near",
    "next to", "in front of", "closest", "farthest", "left of", "right of",
    "to the left", "to the right", "under", "on top of",
]
ORDINAL = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "from the left", "from the right",
]


def has_spatial(q):
    return any(w in q.lower() for w in SPATIAL)


def has_ordinal(q):
    return any(w in q.lower() for w in ORDINAL)


print()
print("=== 1. 逗号细分 ===")
comma_qs = {k: q for k, q in queries.items() if "," in q}
print("含逗号:", len(comma_qs))
n_tight = sum(1 for q in comma_qs.values() if re.search(r"\w,", q))
n_comma_space = sum(1 for q in comma_qs.values() if re.search(r",\s", q))
n_both = sum(1 for q in comma_qs.values() if re.search(r"\w,", q) and re.search(r",\s", q))
print("  逗号紧贴词尾:", n_tight, " 逗号后空格:", n_comma_space, " 两者都有:", n_both)
words = Counter()
for q in comma_qs.values():
    for w in q.split():
        if w.endswith(","):
            words[w.rstrip(",")] += 1
print("  紧贴词尾去重词数:", len(words))
ov_sp = sum(1 for q in comma_qs.values() if has_spatial(q))
ov_ord = sum(1 for q in comma_qs.values() if has_ordinal(q))
print("  逗号 ∩ 空间关系:", ov_sp, " 逗号 ∩ 序数:", ov_ord)

print()
print("=== 2. 撇号细分 ===")
apos_qs = {k: q for k, q in queries.items() if chr(39) in q}
print("含撇号:", len(apos_qs))
n_poss = sum(1 for q in apos_qs.values() if re.search(r"\w's\b", q))
n_quote = sum(1 for q in apos_qs.values() if re.search(r"'\w+'", q))
n_contr = sum(1 for q in apos_qs.values() if re.search(r"\w(n't|'re|'ll|'ve|'d|'m)\b", q))
print("  所有格 x's:", n_poss, " 引号包裹 'x':", n_quote, " 缩写:", n_contr)

print()
print("=== 3. 括号细分 ===")
paren_qs = {k: q for k, q in queries.items() if "(" in q or ")" in q}
print("含括号:", len(paren_qs))
for k, q in list(paren_qs.items()):
    m = re.findall(r"\(([^)]*)\)", q)
    print("  ", k, "|", repr(q[:75]), "->", m)

print()
print("=== 4. 中文夹带 ===")
cjk_qs = {k: q for k, q in queries.items() if re.search(r"[\u4e00-\u9fff]", q)}
print("中文夹带:", len(cjk_qs))
cjk_lens = [
    len(m)
    for q in cjk_qs.values()
    for m in re.findall(r"'([^']*[\u4e00-\u9fff][^']*)'", q)
]
print("  中文引号内容长度:", cjk_lens)

print()
print("=== 5. 缺陷严重度汇总 ===")
levels = [
    ("高", "逗号 749（子集/顺序匹配全失配，退化为 top1）"),
    ("中", "撇号 318（所有格/引号词进集合，标签无撇号时失配）"),
    ("低", "括号 9（(x 与 x) 成独立词）"),
    ("极低", "U+2011 连字符 1 · 内嵌句点 1 · 中文夹带 27（目标词英文可检测）"),
]
for lv, desc in levels:
    print(f"  [{lv}] {desc}")

print()
print("=== 6. 逗号 ∩ 空间约束（清洗可顺带修复） ===")
n_frl = sum(1 for q in comma_qs.values() if "from left to right" in q.lower())
print("  含逗号且含 from left to right:", n_frl)
n_first = sum(
    1
    for q in comma_qs.values()
    if re.search(r",\s+the\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", q.lower())
)
print("  逗号后接序数词:", n_first)
for k, q in comma_qs.items():
    if re.search(r",\s+the\s+(first|second|third|fourth|fifth)\b", q.lower()):
        print("    ", k, "|", repr(q[:90]))
        break
