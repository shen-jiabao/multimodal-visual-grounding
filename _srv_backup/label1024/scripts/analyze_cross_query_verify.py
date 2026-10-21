# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re

STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "with", "and", "to", "from", "at", "is",
    "are", "by", "for", "that", "this", "it", "its", "as", "or", "be", "was",
    "there", "which", "who", "his", "her", "their", "has", "have", "having",
    "not", "no", "all", "also", "than", "over", "under", "near", "beside",
}
ORDINAL_WORDS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth",
}
ORDINAL_RANK = {w: i + 1 for i, w in enumerate(
    ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
     "eighth", "ninth", "tenth", "eleventh", "twelfth"])}


def tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if t.isalpha() and t not in STOPWORDS}


def direction_of(text: str) -> int:
    low = text.lower()
    if "left to right" in low or "left-to-right" in low:
        return 1
    if "right to left" in low or "right-to-left" in low:
        return -1
    return 0


def rank_of(text: str) -> int:
    for w, r in ORDINAL_RANK.items():
        if re.search(rf"\b{w}\b", text.lower()):
            return r
    return 0


def box_center_x(box) -> float:
    return (box[0] + box[2]) / 2.0


def iou(a, b) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="数据集根目录（相对项目根目录）")
    ap.add_argument("--predictions", required=True, help="预测结果 JSON 路径")
    ap.add_argument("--json", default=None, help="导出统计 JSON 的路径（可选）")
    args = ap.parse_args()

    base = os.path.abspath(args.root)
    with open(os.path.join(base, "queries", "queries.json"), encoding="utf-8") as f:
        q = json.load(f)
    with open(args.predictions, encoding="utf-8") as f:
        pred = json.load(f)

    groups: dict[str, list[tuple[str, str, list[float]]]] = {}
    for qid, v in q.items():
        if qid not in pred or not pred[qid]:
            continue
        groups.setdefault(v["visible"], []).append((qid, v["query"], pred[qid]))

    chain_stats = {"groups": 0, "rank_invalid_groups": 0, "direction_ok_groups": 0,
                   "ordered_groups": 0, "violated_groups": 0, "chain_queries": 0}
    chain_examples = []
    same_target = {"pairs": 0, "mean_iou": 0.0, "le30": 0, "le50": 0,
                   "gt70": 0}
    same_target_examples = []

    for vis, items in groups.items():
        if len(items) < 2:
            continue
        toks = [(qid, query, tokenize(query), box) for qid, query, box in items]
        ranked = [(qid, query, box) for qid, query, box, t in
                  [(a[0], a[1], a[3], a[2]) for a in toks]
                  if rank_of(query) > 0]
        if len(ranked) < 2:
            continue
        ranked = sorted(ranked, key=lambda x: rank_of(x[1]))
        ranks = [rank_of(x[1]) for x in ranked]
        k = len(ranked)
        rank_valid = sorted(set(ranks)) == list(range(1, k + 1))
        for a in ranked:
            chain_stats["chain_queries"] += 1
        chain_stats["groups"] += 1
        if not rank_valid:
            chain_stats["rank_invalid_groups"] += 1
            continue
        dirs = [direction_of(x[1]) for x in ranked]
        if all(d != 0 for d in dirs) and len(set(dirs)) == 1:
            d = dirs[0]
            chain_stats["direction_ok_groups"] += 1
            xs = [box_center_x(x[2]) for x in ranked]
            ordered = all((xs[i + 1] - xs[i]) * d > 0 for i in range(len(xs) - 1))
            if ordered:
                chain_stats["ordered_groups"] += 1
            else:
                chain_stats["violated_groups"] += 1
                chain_examples.append({
                    "scene": os.path.basename(vis),
                    "queries": [(x[0], x[1], [round(v, 4) for v in x[2]])
                                for x in ranked],
                    "x_centers": [round(box_center_x(x[2]), 4) for x in ranked],
                    "direction": "L2R" if d == 1 else "R2L",
                })
        elif all(d != 0 for d in dirs):
            chain_stats["direction_ok_groups"] += 1
            chain_stats["violated_groups"] += 1

        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                qa, ta, boxa = toks[i][1], toks[i][2], toks[i][3]
                qb, tb, boxb = toks[j][1], toks[j][2], toks[j][3]
                inter = ta & tb
                union = ta | tb
                if not union:
                    continue
                jac = len(inter) / len(union)
                if jac >= 0.5 and rank_of(qa) == 0 and rank_of(qb) == 0:
                    v = iou(boxa, boxb)
                    same_target["pairs"] += 1
                    same_target["mean_iou"] += v
                    if v <= 0.3:
                        same_target["le30"] += 1
                    if v <= 0.5:
                        same_target["le50"] += 1
                    if v > 0.7:
                        same_target["gt70"] += 1
                    if len(same_target_examples) < 10:
                        same_target_examples.append({
                            "scene": os.path.basename(vis),
                            "a": [qa, [round(x, 4) for x in boxa]],
                            "b": [qb, [round(x, 4) for x in boxb]],
                            "jaccard": round(jac, 3),
                            "iou": round(v, 3),
                        })

    if same_target["pairs"]:
        same_target["mean_iou"] = round(same_target["mean_iou"] / same_target["pairs"], 4)
        same_target["le30_ratio"] = round(same_target["le30"] / same_target["pairs"], 4)
        same_target["le50_ratio"] = round(same_target["le50"] / same_target["pairs"], 4)
        same_target["gt70_ratio"] = round(same_target["gt70"] / same_target["pairs"], 4)

    stats = {"chain": chain_stats, "same_target": same_target,
             "chain_examples": chain_examples[:5],
             "same_target_examples": same_target_examples}
    print("序数链:", json.dumps(chain_stats, ensure_ascii=False))
    print("同目标对:", json.dumps(same_target, ensure_ascii=False))
    print("\n序数链违反示例:")
    for ex in chain_examples[:5]:
        print("  ", json.dumps(ex, ensure_ascii=False)[:400])
    print("\n同目标对示例:")
    for ex in same_target_examples[:5]:
        print("  ", json.dumps(ex, ensure_ascii=False)[:300])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"\n已导出: {args.json}")


if __name__ == "__main__":
    main()
