"""错误分布诊断脚本（方案 9）：互斥分桶统计 + 横切指标。

运行：
    python scripts/初赛/analyze_query_patterns.py --predictions outputs/predictions.json \
        --ground_truth scripts/初赛/annotation_tool/outputs/annotations/annotations.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from src.config import load_config
from src.eval.metrics import iou
from src.inference.matcher import _parse_target_container, parse_spatial_constraint
from src.utils.io import load_json

SMALL_AREA = 0.01


def bucket_of(query: str, cons: dict | None, gt_box: list[float]) -> str:
    if cons and cons.get("kind") == "depth":
        return "depth"
    if cons:
        return "ordinal"
    if _parse_target_container(query)[0] is not None:
        return "container"
    x1, y1, x2, y2 = gt_box
    if (x2 - x1) * (y2 - y1) < SMALL_AREA:
        return "small"
    return "generic"


def main() -> None:
    p = argparse.ArgumentParser(description="错误分布诊断")
    p.add_argument("--config", default=None)
    p.add_argument("--predictions", required=True)
    p.add_argument("--ground_truth", required=True)
    p.add_argument("--queries", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    queries = load_json(args.queries or cfg.queries_path)
    preds = load_json(args.predictions)
    gts = load_json(args.ground_truth)

    common = sorted(set(preds) & set(gts))
    buckets: dict[str, list[dict]] = {}
    cross = {"none": 0, "oversized": 0, "undersized": 0, "total_iou": 0.0}
    for qid in common:
        gt_box = gts[qid].get("bbox") if isinstance(gts[qid], dict) else gts[qid]
        if not gt_box:
            continue
        query = queries[qid]["query"] if isinstance(queries[qid], dict) else queries[qid]
        cons = parse_spatial_constraint(query)
        key = bucket_of(query, cons, gt_box)
        pred = preds[qid]
        buckets.setdefault(key, []).append({"qid": qid, "pred": pred, "gt": gt_box})
        if pred is None:
            cross["none"] += 1
            continue
        cross["total_iou"] += iou(pred, gt_box)
        pa = (pred[2] - pred[0]) * (pred[3] - pred[1])
        ga = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        if ga > 0 and pa >= 2 * ga:
            cross["oversized"] += 1
        if ga > 0 and pa <= 0.5 * ga:
            cross["undersized"] += 1

    n_total = len(common)
    n_correct = sum(
        1 for qid in common
        if preds[qid] is not None and iou(preds[qid], gts[qid]["bbox"]) >= 0.5
    )
    report = {
        "num_total": n_total,
        "acc": n_correct / n_total if n_total else float("nan"),
        "mean_iou": cross["total_iou"] / n_total if n_total else float("nan"),
        "num_none": cross["none"],
        "oversized": cross["oversized"],
        "undersized": cross["undersized"],
        "buckets": {},
    }
    for key, items in sorted(buckets.items()):
        acc = sum(1 for it in items if it["pred"] is not None and iou(it["pred"], it["gt"]) >= 0.5)
        report["buckets"][key] = {
            "num": len(items),
            "acc": acc / len(items),
            "num_correct": acc,
        }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
