# -*- coding: utf-8 -*-
"""上云前审计：本轮每个变量各会碰到多少条 query（只读，不需要云端数据）。

多变量同轮的前提是「每个变量事后都能归因」，而归因的第一件事是先知道
**受影响集合有多大**：

  · 只碰几十条的变量，不可能解释整包的 ±0.01，整包退化时也不必先怀疑它；
  · 碰上千条的变量，必须事先准备好它的反事实证据。

本脚本只用本地 `queries.json` + 当前代码，量出三个**代码级**变量（没有配置
开关，随代码一起生效）的受影响集合：

  D  提示词清洗 ``clean_prompt_text``：哪些 query 的提示词文本被改写了？
  E  同文本补齐 ``propagate_identical_queries``：同图逐字相同的 query 有多少？
  F  约束解析 ``parse_spatial_constraint``：新增覆盖分别来自「放宽桥」与
     「黑名单词边界」，各多少条 —— 这是它可能移动有效框的规模上限。

配置级变量（``max_new_tokens`` / ``multi_box_id`` / ``enable_depth_prior``）
不在本地判定，其归因方案见 docs/说明/云端运行说明书-多变量同轮.md。

用法：
    python scripts/复赛/round_audit.py
    python scripts/复赛/round_audit.py --json outputs/复赛/audit/round_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import matcher as M          # noqa: E402
from src.utils.text import clean_prompt_text    # noqa: E402

ORD_PAT = "|".join(M._ORDINALS)
# 与 matcher.parse_spatial_constraint 里的两条桥逐字一致（原式 / 放宽式）
STRICT_RE = re.compile(
    r"\b(" + ORD_PAT + r")\b\s+(?:\w+\s+)*?from\s+(?:the\s+)?(left|right|top|bottom)")
RELAXED_RE = re.compile(
    r"\b(" + ORD_PAT + r")\b[\s\w,.'\-]{0,90}?from\s+(?:the\s+)?(left|right|top|bottom)")


def strip_blacklist(text: str) -> str:
    """复刻 parse_spatial_constraint 的黑名单预处理（当前 = 词边界版）。"""
    for blocked in M._FIRST_BLACKLIST:
        text = re.sub(r"\bfirst\s+" + blocked + r"\b", "", text)
    return text


def blacklist_boundary_hits(norm_text: str) -> int:
    """旧实现（无 \\b）会误删、当前实现不会的处数：`first <blocked>` 后面还接词。

    典型：``the first handrail from the left`` —— 旧式朴素子串会把 ``first hand``
    吃掉，剩 ``rail from the left``，序数丢失、约束静默降级为 None。
    """
    hits = 0
    for blocked in M._FIRST_BLACKLIST:
        for m in re.finditer(r"first\s+" + blocked, norm_text):
            end = m.end()
            if end < len(norm_text) and (norm_text[end].isalnum()
                                        or norm_text[end] == "_"):
                hits += 1
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description="上云前变量受影响集合审计")
    ap.add_argument("--root", default="dataset", help="数据根（含 queries/queries.json）")
    ap.add_argument("--json", default=None, help="导出审计结果")
    ap.add_argument("--top", type=int, default=5, help="每节展示的示例条数")
    args = ap.parse_args()

    qpath = os.path.join(os.path.abspath(args.root), "queries", "queries.json")
    with open(qpath, encoding="utf-8") as fh:
        qraw = json.load(fh)
    total = len(qraw)
    print("查询总数 %d  (%s)" % (total, qpath))

    # ---------------- D 提示词清洗 ----------------
    d_changed = [k for k, v in qraw.items()
                 if clean_prompt_text(v["query"]) != v["query"]]
    print("\n=== D 提示词清洗 clean_prompt_text ===")
    print("  文本被改写的 query：%d  (%.2f%%)" % (len(d_changed), 100.0 * len(d_changed) / total))
    for k in d_changed[:args.top]:
        print("    %s  %s" % (k, repr(qraw[k]["query"])[:96]))
        print("       -> %s" % repr(clean_prompt_text(qraw[k]["query"]))[:96])

    # ---------------- E 同图同文本 ----------------
    groups = defaultdict(list)
    for k, v in qraw.items():
        groups[(v["visible"], clean_prompt_text(v["query"]))].append(k)
    dup = {key: ks for key, ks in groups.items() if len(ks) > 1}
    dup_queries = sum(len(ks) for ks in dup.values())
    dup_redundant = sum(len(ks) - 1 for ks in dup.values())
    print("\n=== E 同文本补齐 propagate_identical_queries ===")
    print("  同图逐字相同的 query 组：%d 组" % len(dup))
    print("  落在这些组里的 query：%d  (%.2f%%)" % (dup_queries, 100.0 * dup_queries / total))
    print("  其中「兄弟已有框、本条可被补齐」的上限：%d 条" % dup_redundant)
    for key in sorted(dup, key=lambda k: -len(dup[k]))[:args.top]:
        print("    %s  ×%d" % (key[1][:88], len(dup[key])))

    # ---------------- F 约束解析 ----------------
    f_relaxed, f_blacklist = [], []
    n_parse = n_axis = 0
    for k, v in qraw.items():
        norm = M.normalize_text(v["query"])
        stripped = strip_blacklist(norm)
        if not STRICT_RE.search(stripped) and RELAXED_RE.search(stripped):
            f_relaxed.append(k)
        if blacklist_boundary_hits(norm):
            f_blacklist.append(k)
        cons = M.parse_spatial_constraint(v["query"])
        if cons:
            n_parse += 1
            if cons.get("kind") != "depth":
                n_axis += 1
    new_cov = set(f_relaxed) | set(f_blacklist)
    print("\n=== F 约束解析 parse_spatial_constraint ===")
    print("  当前解析出约束：%d  (%.2f%%)，其中非深度（会排序取第 k 个）：%d"
          % (n_parse, 100.0 * n_parse / total, n_axis))
    print("  新增覆盖 · 放宽桥（原式不中）：%d" % len(f_relaxed))
    print("  新增覆盖 · 黑名单词边界：    %d" % len(f_blacklist))
    print("  新增覆盖合计（去重）：      %d  (%.2f%%)"
          % (len(new_cov), 100.0 * len(new_cov) / total))
    print("  → 这 %d 条是 F 可能改变框的规模上限；真正移动还需「同类候选 > 1」。" % len(new_cov))
    for k in sorted(new_cov)[:args.top]:
        print("    %s  %s" % (k, qraw[k]["query"][:92]))

    # ---------------- 汇总 ----------------
    print("\n=== 汇总（代码级变量） ===")
    print("  %-6s %-34s %8s %9s" % ("变量", "含义", "受影响", "占比"))
    rows = [
        ("D", "提示词清洗（改写文本）", len(d_changed)),
        ("E", "同文本补齐（可补条数）", dup_redundant),
        ("F", "约束解析新增覆盖（上限）", len(new_cov)),
    ]
    for tag, desc, n in rows:
        print("  %-6s %-34s %8d %8.2f%%" % (tag, desc, n, 100.0 * n / total))

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        payload = {
            "total": total,
            "D": {"changed": len(d_changed), "qids": d_changed},
            "E": {"groups": len(dup), "queries": dup_queries,
                  "redundant": dup_redundant,
                  "group_sizes": sorted((len(v) for v in dup.values()), reverse=True)},
            "F": {"parse_total": n_parse, "parse_axis": n_axis,
                  "relaxed_bridge": len(f_relaxed), "blacklist_boundary": len(f_blacklist),
                  "new_coverage": len(new_cov), "qids": sorted(new_cov)},
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n已导出：" + args.json)


if __name__ == "__main__":
    main()
