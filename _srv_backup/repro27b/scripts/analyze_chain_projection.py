# -*- coding: utf-8 -*-
"""序数链修正的离线 A/B：在**已有预测**上量测单调性修正的收益（无需 GT）。

口径与 scripts/analyze_ordinal_chains.py 完全一致（同 residual_tokens / classify_chain），
因此 before 的 A_ratio 应与 ordinal_chains_20260927.json 完全对齐（0.7027），
after 则是施加 enforce_ordinal_chains 之后的同一指标。

用法：
    python scripts/analyze_chain_projection.py \
        --predictions outputs/复赛/predictions_qwen_multiimg.json \
        --mode sort --gap-ratio 0.0 --json outputs/复赛/chain/chain_projection_20260927.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.inference.chains import (
    classify_chain,
    enforce_ordinal_chains,
    residual_tokens,
)
from src.inference.matcher import parse_spatial_constraint


def build_groups(q, pred, key_fn):
    """按 (visible, key) 归链；返回 [(scene, key, items)]，items=[(qid, query, box, cons)]。"""
    groups: dict[tuple, list] = {}
    for qid, v in q.items():
        if qid not in pred or not pred[qid]:
            continue
        cons = parse_spatial_constraint(v["query"])
        if cons is None or cons.get("kind") == "depth" or cons.get("rank", 0) <= 0:
            continue
        groups.setdefault((v["visible"], key_fn(v["query"])), []).append(
            (qid, v["query"], list(pred[qid]), cons))
    return [(vis, key, items) for (vis, key), items in groups.items()]


def measure(groups):
    counts = {"groups": 0, "queries": 0, "skip_axis": 0, "skip_rank": 0,
              "valid": 0, "A": 0, "ordered": 0, "B_dup": 0, "B_tie": 0,
              "B_broken": 0}
    for _vis, _key, items in groups:
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
    denom = counts["A"] + counts["B_dup"] + counts["B_tie"] + counts["B_broken"]
    return counts, round(counts["A"] / max(1, denom), 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--mode", default="sort", choices=["sort", "project"])
    ap.add_argument("--gap-ratio", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = os.path.abspath(args.root)
    with open(os.path.join(base, "queries", "queries.json"), encoding="utf-8") as f:
        q = json.load(f)
    with open(args.predictions, encoding="utf-8") as f:
        pred = json.load(f)

    key_fn = residual_tokens
    before_groups = build_groups(q, pred, key_fn)
    counts_b, ratio_b = measure(before_groups)

    # ---- 逐场景施加序数链修正 ----
    scenes: dict[str, list[str]] = {}
    for qid, v in q.items():
        scenes.setdefault(v["visible"], []).append(qid)

    pred_after = dict(pred)
    deltas = []
    for _vis, qids in scenes.items():
        qids = [x for x in qids if x in pred and pred[x]]
        queries = [q[x]["query"] for x in qids]
        boxes = [list(pred[x]) for x in qids]
        fixed = enforce_ordinal_chains(queries, boxes, mode=args.mode,
                                       gap_ratio=args.gap_ratio)
        for x, b0, b1 in zip(qids, boxes, fixed):
            if b1 is None:
                continue
            pred_after[x] = b1
            d = abs((b1[0] + b1[2]) / 2 - (b0[0] + b0[2]) / 2) + \
                abs((b1[1] + b1[3]) / 2 - (b0[1] + b0[3]) / 2)
            if d > 1e-9:
                deltas.append(d)

    after_groups = build_groups(q, pred_after, key_fn)
    counts_a, ratio_a = measure(after_groups)

    print("before:", json.dumps(counts_b, ensure_ascii=False), "A_ratio=", ratio_b)
    print("after :", json.dumps(counts_a, ensure_ascii=False), "A_ratio=", ratio_a)
    print(f"单调率 ordered/valid: {counts_b['ordered']}/{counts_b['valid']}"
          f" = {counts_b['ordered'] / max(1, counts_b['valid']):.4f}"
          f"  ->  {counts_a['ordered']}/{counts_a['valid']}"
          f" = {counts_a['ordered'] / max(1, counts_a['valid']):.4f}")
    print(f"被移动的预测数: {len(deltas)} / {sum(1 for v in pred.values() if v)}"
          f"  平均位移={float(np.mean(deltas)) if deltas else 0.0:.4f}"
          f"  最大位移={float(np.max(deltas)) if deltas else 0.0:.4f}")

    if args.json:
        stats = {"mode": args.mode, "gap_ratio": args.gap_ratio,
                 "before": {"counts": counts_b, "A_ratio": ratio_b},
                 "after": {"counts": counts_a, "A_ratio": ratio_a},
                 "moved": len(deltas),
                 "mean_shift": float(np.mean(deltas)) if deltas else 0.0,
                 "max_shift": float(np.max(deltas)) if deltas else 0.0}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print("已导出:", args.json)


if __name__ == "__main__":
    main()