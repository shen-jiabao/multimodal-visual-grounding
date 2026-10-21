# -*- coding: utf-8 -*-
"""查询文本卫生审计（只读）。

审查 5690 条复赛 query 里可能让模型「读不懂 / 答偏 / 回显不了」的文本特征，
并量化两件事：

  1. **送进提示词之前** query 有没有带噪声（BOM / 零宽 / 异体连字符 / 弯引号 /
     换行 / 不换行空格 / 控制字符）——`clean_query` 已实现，但提示词构造
     （`build_multi_prompt`）用的是**原始** `s.query`，清洗从未生效；
  2. 约束解析（`parse_spatial_constraint`）漏掉了哪些**明显**带约束的 query，
     以及漏掉的成因。典型嫌疑：序数与 "from left to right" 之间夹了逗号，
     被 `(?:\\w+\\s+)*?` 这座「桥」挡住。

全部只读，不加载模型、不改任何预测。

用法::

    python scripts/复赛/diag_query_hygiene.py
    python scripts/复赛/diag_query_hygiene.py --json outputs/复赛/diag/query_hygiene.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.matcher import parse_spatial_constraint  # noqa: E402
from src.utils.text import clean_query  # noqa: E402

# ---------------------------------------------------------------- 词表
ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth")
ORD_PAT = "|".join(ORDINALS)

# 现状：序数与 from 之间只允许「纯单词 + 空白」
RE_BRIDGE_OLD = re.compile(
    rf"\b({ORD_PAT})\b\s+(?:\w+\s+)*?from\s+(?:the\s+)?(left|right|top|bottom)")
# 放宽：允许逗号 / 撇号 / 连字符 / 句点在桥中间（真正的定语会带这些）
RE_BRIDGE_NEW = re.compile(
    rf"\b({ORD_PAT})\b[\s\w,.'\-]{{0,90}}?from\s+(?:the\s+)?(left|right|top|bottom)")

# 明显是约束、但现状解析器未必覆盖的说法
LEXICON = {
    "left/right 短语（非 most）": r"\b(on|to)\s+the\s+(left|right)\b|\b(left|right)\s+(one|side)\b",
    "last / final": r"\b(last|final)\b",
    "middle / center": r"\b(middle|center|centre|centered)\b",
    "penultimate / second to last": r"\b(penultimate|second\s+to\s+last|second-to-last)\b",
    "11th..20th": r"\b(eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth)\b",
    "top / bottom (裸词)": r"\b(top|bottom)\b",
    "foreground / background": r"\b(foreground|background)\b",
    "depth 变体": r"\b(furthest|farthest|most\s+distant|nearest|closest)\s*(away)?\b",
}

# 提示词结构风险：这些字符会破坏 "Targets:" 的 `N. <query>` 单行列表
STRUCT_HAZARDS = {
    "含换行": r"[\r\n]",
    "含制表符": r"\t",
    "含双引号": r'"',
    "含花括号": r"[{}]",
    "含方括号": r"[\[\]]",
    "疑似注入（出现指令词）": r"(?i)\b(return only|json|targets?:|ignore (the )?(previous|above))\b",
}


# ---------------------------------------------------------------- 建议的清洗
_CURLY = (("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'),
          ("\u2026", "..."), ("\u00a0", " "), ("\u3000", " "))


def clean_prompt_text(text: str) -> str:
    """提示词侧建议的 query 清洗：在 `clean_query` 之上补齐排版类噪声。

    只做「等价改写」——不改词序、不改语义、不改大小写，因此不改变任何一条
    query 的含义，只把模型需要额外消化的噪声字符去掉。
    """
    t = clean_query(text)                       # BOM / 零宽 / 连字符族
    for a, b in _CURLY:
        t = t.replace(a, b)
    t = re.sub(r"[\r\n\t\f\v]+", " ", t)        # 换行/制表 → 空格
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def is_control(ch: str) -> bool:
    return unicodedata.category(ch) == "Cc" and ch not in "\t\n\r"


# ---------------------------------------------------------------- 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="查询文本卫生审计（只读）")
    ap.add_argument("--queries", default="dataset/queries/queries.json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as fh:
        raw = json.load(fh)
    items = list(raw.items())
    total = len(items)

    rep: dict = {"total": total}
    ex = args.examples

    # ---------------- A. 字符卫生 ----------------
    changed_by_clean_query, changed_by_prompt = [], []
    char_census = collections.Counter()
    ctrl_examples = []
    for qid, it in items:
        q = it["query"]
        if clean_query(q) != q:
            changed_by_clean_query.append((qid, q))
        if clean_prompt_text(q) != q:
            changed_by_prompt.append((qid, q))
        for ch in set(q):
            if ord(ch) > 126 or is_control(ch):
                char_census[f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}"] += 1
        if any(is_control(c) for c in q):
            ctrl_examples.append((qid, q))

    rep["char_census"] = dict(char_census.most_common(15))
    rep["n_changed_by_clean_query"] = len(changed_by_clean_query)
    rep["n_changed_by_clean_prompt"] = len(changed_by_prompt)
    rep["control_char_queries"] = len(ctrl_examples)
    print("=" * 78)
    print("A. 字符卫生（送到提示词之前的噪声）")
    print(f"  总 query {total}")
    print(f"  `clean_query` 会改写        {len(changed_by_clean_query):4d} 条"
          f"   ({100.0 * len(changed_by_clean_query) / total:.2f}%)")
    print(f"  建议的 `clean_prompt_text`  {len(changed_by_prompt):4d} 条"
          f"   ({100.0 * len(changed_by_prompt) / total:.2f}%)  ← 增量 "
          f"{len(changed_by_prompt) - len(changed_by_clean_query)} 条")
    print(f"  含控制字符（\\r\\n\\t 之外）   {len(ctrl_examples)} 条")
    print("  非 ASCII / 控制字符 census：")
    for k, c in char_census.most_common(12):
        print(f"      {k:<46s} {c:5d} 条")
    if changed_by_clean_query[:ex]:
        print("  `clean_query` 改写样例：")
        for qid, q in changed_by_clean_query[:ex]:
            print(f"      {qid}  {q[:70]!r}")
            print(f"      {'':>10s}→ {clean_query(q)[:70]!r}")
    if changed_by_prompt[:ex]:
        print("  建议清洗**新增**的样例：")
        for qid, q in changed_by_prompt[:ex]:
            print(f"      {qid}  {q[:70]!r}")
            print(f"      {'':>10s}→ {clean_prompt_text(q)[:70]!r}")

    # ---------------- B. 提示词结构风险 ----------------
    struct = collections.Counter()
    struct_ex = collections.defaultdict(list)
    for qid, it in items:
        q = it["query"]
        for name, pat in STRUCT_HAZARDS.items():
            if re.search(pat, q):
                struct[name] += 1
                if len(struct_ex[name]) < ex:
                    struct_ex[name].append((qid, q))
    rep["struct_hazards"] = dict(struct)
    print("\n" + "=" * 78)
    print("B. 提示词结构风险（会破坏 `N. <query>` 的单行列表 / 诱导模型跑题）")
    if not struct:
        print("  无")
    for name, c in struct.most_common():
        print(f"  {name:<26s} {c:5d} 条")
        for qid, q in struct_ex[name][:2]:
            print(f"      {qid}  {q[:78]!r}")

    # ---------------- C. 同图重复 query ----------------
    by_scene = collections.defaultdict(list)
    for qid, it in items:
        by_scene[qid.split("_")[0]].append((qid, it["query"]))
    dup_scenes, dup_queries, dup_groups = 0, 0, 0
    dup_examples = []
    for scene, lst in by_scene.items():
        cnt = collections.Counter(q for _, q in lst)
        d = {q: c for q, c in cnt.items() if c >= 2}
        if d:
            dup_scenes += 1
            dup_queries += sum(d.values())
            dup_groups += len(d)
            if len(dup_examples) < ex:
                q0 = next(iter(d))
                dup_examples.append((scene, [qid for qid, q in lst if q == q0][:4], q0))
    rep["dup_within_image"] = {"scenes": dup_scenes, "groups": dup_groups,
                               "queries": dup_queries}
    print("\n" + "=" * 78)
    print("C. 同图**文本完全相同**的 query（ID 化提示词下模型可能只回一条）")
    print(f"  涉及 {dup_scenes} 张图 / {dup_groups} 组 / {dup_queries} 条 query"
          f"  ({100.0 * dup_queries / total:.2f}%)")
    for scene, qids, q0 in dup_examples:
        print(f"      {scene}  {qids}  {q0[:64]!r}")

    # ---------------- D. 约束解析覆盖 ----------------
    parsed = collections.Counter()
    unparsed_but_lexicon = collections.Counter()
    for qid, it in items:
        cons = parse_spatial_constraint(it["query"])
        parsed["有约束" if cons else "无约束"] += 1
        if not cons:
            for name, pat in LEXICON.items():
                if re.search(pat, it["query"], re.I):
                    unparsed_but_lexicon[name] += 1

    hit_old = [qid for qid, it in items if RE_BRIDGE_OLD.search(it["query"])]
    hit_new = [qid for qid, it in items if RE_BRIDGE_NEW.search(it["query"])]
    only_new = [q for q in hit_new if q not in set(hit_old)]
    bridge_ex = [(q, raw[q]["query"]) for q in only_new[:ex]]

    rep["constraint"] = {
        "parsed": parsed["有约束"], "unparsed": parsed["无约束"],
        "coverage_pct": round(100.0 * parsed["有约束"] / total, 2),
        "bridge_old": len(hit_old), "bridge_new": len(hit_new),
        "bridge_gain": len(only_new),
        "unparsed_lexicon": dict(unparsed_but_lexicon.most_common()),
    }
    print("\n" + "=" * 78)
    print("D. 约束解析覆盖（`parse_spatial_constraint`）")
    print(f"  现覆盖 {parsed['有约束']} / {total} = "
          f"{100.0 * parsed['有约束'] / total:.2f}%")
    print("  未解析但词面明显带约束的：")
    for name, c in unparsed_but_lexicon.most_common():
        print(f"      {name:<30s} {c:5d} 条")
    print("\n  ★ 序数 ↔ `from left/right` 的「桥」")
    print(f"      现状正则 `(?:\\w+\\s+)*?`  命中 {len(hit_old):4d} 条")
    print(f"      容许逗号/连字符/撇号     命中 {len(hit_new):4d} 条   "
          f"净增 {len(only_new)} 条")
    for qid, q in bridge_ex:
        print(f"      + {qid}  {q[:74]!r}")

    # ---------------- E. 结论 ----------------
    print("\n" + "=" * 78)
    print("E. 结论")
    if len(changed_by_prompt):
        print(f"  · 提示词侧清洗：{len(changed_by_prompt)} 条 query 带着噪声进模型，"
              f"其中 {len(changed_by_prompt) - len(changed_by_clean_query)} 条是"
              f"`clean_query` 也没覆盖的（换行/弯引号/不换行空格）。")
    if len(only_new):
        print(f"  · 约束解析：逗号桥被挡掉 {len(only_new)} 条，修正是**超集**，"
              f"不可能让任何原本解析成功的 query 失败。")
    print("  · 注意：当前 `multi_box_id=true` 下每条 query 只对应一个框，"
          "`match_by_position` 的约束分支（len(cands)>1）不会触发——")
    print("    故约束解析的修正**对当前基线是惰性的**，价值在 detect_select / "
          "多候选路径。")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=1)
        print(f"\n已导出：{args.json}")


if __name__ == "__main__":
    main()