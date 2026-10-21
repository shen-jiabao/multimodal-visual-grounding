"""container 桶错误归因诊断（8.1 §8.2 第一步）。

对 container 桶查询（命中容器结构且非 depth/ordinal）按错误类型分桶：
correct / oversized(>=2x) / undersized(<=0.5x) / misplaced(中心距>0.1) / missed(None 或占位框)。
输出统计与每类典型样本，供人工抽框确认主模式。

用法：
    python scripts/初赛/diagnose_container.py --predictions <P> --ground_truth <GT> \
        --queries dataset/queries/queries.json --output <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from src.eval.metrics import iou
from src.inference.matcher import _parse_target_container, parse_spatial_constraint
from src.utils.io import load_json

PLACEHOLDER = [0.49, 0.49, 0.51, 0.51]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center_dist(a: list[float], b: list[float]) -> float:
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def classify(pred: list[float] | None, gt: list[float]) -> tuple[str, float]:
    if pred is None or pred == PLACEHOLDER:
        return "missed", 0.0
    ov = iou(pred, gt)
    if ov >= 0.5:
        return "correct", ov
    pa, ga = area(pred), area(gt)
    if ga > 0 and pa >= 2 * ga:
        return "oversized", ov
    if ga > 0 and pa <= 0.5 * ga:
        return "undersized", ov
    if center_dist(pred, gt) > 0.1:
        return "misplaced", ov
    return "misplaced", ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--ground_truth", required=True)
    ap.add_argument("--queries", default="dataset/queries/queries.json")
    ap.add_argument("--output", required=True)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--full-out", default=None)
    args = ap.parse_args()

    preds = load_json(args.predictions)
    gts = load_json(args.ground_truth)
    queries = load_json(args.queries)

    groups: dict[str, list[dict]] = {}
    for qid in sorted(set(preds) & set(gts)):
        g = gts[qid]
        gt = g.get("bbox") if isinstance(g, dict) else g
        if not gt:
            continue
        q = queries[qid]["query"] if isinstance(queries[qid], dict) else queries[qid]
        cons = parse_spatial_constraint(q)
        if cons:
            continue
        if _parse_target_container(q)[0] is None:
            continue
        kind, ov = classify(preds[qid], gt)
        groups.setdefault(kind, []).append(
            {"qid": qid, "query": q, "pred": preds[qid], "gt": gt, "iou": round(ov, 4)}
        )

    report = {"num_total": sum(len(v) for v in groups.values()), "buckets": {}}
    for kind in ("correct", "oversized", "undersized", "misplaced", "missed"):
        items = groups.get(kind, [])
        report["buckets"][kind] = {
            "num": len(items),
            "pct": round(100 * len(items) / max(1, report["num_total"]), 2),
        }
        if items:
            report["buckets"][kind]["top"] = sorted(
                items, key=lambda x: x["iou"] if kind == "correct" else -x["iou"]
            )[: args.topk]

    print("container 桶总数:", report["num_total"])
    for k, v in report["buckets"].items():
        print(f"  {k:10s} n={v['num']:5d}  {v['pct']:.2f}%")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if args.full_out:
        with open(args.full_out, "w", encoding="utf-8") as f:
            json.dump({k: {"num": len(v), "items": v} for k, v in groups.items()}, f, ensure_ascii=False)
        print("full ->", args.full_out)
    print("saved ->", args.output)


if __name__ == "__main__":
    main()
