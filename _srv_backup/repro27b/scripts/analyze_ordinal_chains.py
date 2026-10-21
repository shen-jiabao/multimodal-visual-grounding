# -*- coding: utf-8 -*-
"""序数链破序统计（离线代理指标，本地无 GT）。

复用 :mod:`src.inference.chains` 的 ``residual_tokens`` / ``classify_chain``：
此处原先有一份重复实现，现已合并，保证统计口径与 ``analyze_chain_projection.py``
的离线 A/B 完全一致。

指标含义：
    ``A`` = 无法用重复/并列/断裂解释的破序（"纯错"，收益上限）；
    ``B_*`` = 可被重复/并列/断裂解释的破序；``ordered`` = 单调（正确）。
    ``A_ratio = A / (A + B_dup + B_tie + B_broken)``，越大越差。

用法：
    python scripts/analyze_ordinal_chains.py \
        --predictions outputs/复赛/predictions_qwen_multiimg.json \
        --json outputs/复赛/chain/ordinal_chains_20260927.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference.chains import classify_chain, residual_tokens
from src.inference.matcher import parse_spatial_constraint


def build_groups(q: dict, pred: dict) -> dict:
    """按 (visible 图, 残余词集合) 归链：items = (qid, query, box, cons)。"""
    groups: dict[tuple, list] = {}
    for qid, v in q.items():
        if qid not in pred or not pred[qid]:
            continue
        cons = parse_spatial_constraint(v["query"])
        if cons is None or cons.get("kind") == "depth" or cons.get("rank", 0) <= 0:
            continue
        groups.setdefault((v["visible"], residual_tokens(v["query"])), []).append(
            (qid, v["query"], list(pred[qid]), cons))
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="数据集根目录（相对项目根目录）")
    ap.add_argument("--predictions", required=True, help="预测 JSON 路径")
    ap.add_argument("--json", default=None, help="导出统计 JSON 的路径（可选）")
    args = ap.parse_args()

    base = os.path.abspath(args.root)
    with open(os.path.join(base, "queries", "queries.json"), encoding="utf-8") as f:
        q = json.load(f)
    with open(args.predictions, encoding="utf-8") as f:
        pred = json.load(f)

    groups = build_groups(q, pred)

    counts = {"groups": 0, "queries": 0, "skip_axis": 0, "skip_rank": 0,
              "valid": 0, "A": 0, "ordered": 0, "B_dup": 0, "B_tie": 0,
              "B_broken": 0}
    examples: dict[str, list] = {"A": [], "B_broken": [], "B_tie": [], "B_dup": []}
    for (vis, _key), items in groups.items():
        if len(items) < 2:
            continue
        counts["groups"] += 1
        counts["queries"] += len(items)
        label = classify_chain(items)
        if label.startswith("skip"):
            counts[label] += 1
            continue
        counts["valid"] += 1
        counts[label] += 1
        if len(examples.get(label, [])) < 4:
            row = {
                "scene": os.path.basename(vis),
                "items": [(x[0], x[1], [round(b, 4) for b in x[2]])
                          for x in sorted(items, key=lambda y: y[3]["rank"])],
            }
            examples.setdefault(label, []).append(row)

    stats = {"counts": counts}
    stats["A_ratio"] = round(counts["A"] / max(1, counts["A"] + counts["B_dup"]
                                               + counts["B_tie"] + counts["B_broken"]), 4)
    stats["examples"] = examples

    print("counts:", json.dumps(counts, ensure_ascii=False))
    print(f"A 占比（越大越差）: {stats['A_ratio']}")
    for label in ("A", "B_broken", "B_tie", "B_dup"):
        print(f"\n[{label}] 示例:")
        for ex in examples.get(label, [])[:3]:
            print("  ", json.dumps(ex, ensure_ascii=False)[:500])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"\n已导出: {args.json}")


if __name__ == "__main__":
    main()