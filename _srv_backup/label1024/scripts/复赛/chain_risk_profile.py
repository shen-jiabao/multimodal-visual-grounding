# -*- coding: utf-8 -*-
"""序数链重指派的风险画像：区分"安全可修"与"高风险"，并扫描 gap_ratio / max_shift。

要回答的问题（离线、无 GT）：
  Q1 那 42 条破序链里，有多少只是"几乎并列"（几何上微调即可，重指派代价低），
     多少是真的把框写到了别处（重指派位移巨大，可能在 ACC@0.5 下把对的框改错）？
  Q2 位移分布长什么样（均值会被少数极端值带偏）。
  Q3 组内方向提示是否互相矛盾（"first ... from right to left" 与
     "second ... from left to right" 同链）——这类矛盾组是重指派最强的理由。

输出：每个配置的位移分位数、破序链的几何分类、矛盾方向组清单（前若干条）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "复赛"))

from proxy_metrics import (  # noqa: E402
    build_chain_groups_v2,
    chain_stats,
    load_predictions,
    quantile,
)
from src.inference.chains import classify_chain, enforce_ordinal_chains  # noqa: E402


def chain_geometry(queries_raw, pred):
    """把破序链按几何特征分类，并统计方向提示矛盾情况。"""
    groups = build_chain_groups_v2(queries_raw, pred)
    cls = Counter()
    detail = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        label = classify_chain(items)
        if label in ("ordered", "skip_axis", "skip_rank"):
            continue
        scene, axis, direction, rkey = key
        axis_i = 0 if axis == "x" else 1
        pairs = [(q, qq, b, c) for q, qq, b, c in items]
        centers = sorted(((b[0] + b[2]) / 2 if axis_i == 0 else (b[1] + b[3]) / 2, q)
                         for q, _qq, b, _c in pairs)
        gaps = [centers[i + 1][0] - centers[i][0] for i in range(len(centers) - 1)]
        span = centers[-1][0] - centers[0][0]
        # 方向提示矛盾：query 文本里出现与组方向相反的 from ... to ...
        import re
        opposite = 0
        for _q, qq, _b, _c in pairs:
            low = qq.lower()
            if direction > 0 and ("right to left" in low or "right-to-left" in low):
                opposite += 1
            if direction < 0 and ("left to right" in low or "left-to-right" in low):
                opposite += 1
        if min(gaps) > 0.15:
            shape = "spread_far"        # 彼此分得很开：重指派 = 大位移
        elif span <= 0.15:
            shape = "near_tie"          # 几乎并列：重指派/微调都便宜
        else:
            shape = "mixed"
        cls[(label, shape)] += 1
        cls["opposite_%d" % (1 if opposite else 0)] += 1
        detail.append({
            "scene": os.path.basename(scene), "label": label, "shape": shape,
            "members": len(items), "span": round(span, 4),
            "min_gap": round(min(gaps), 4), "max_gap": round(max(gaps), 4),
            "opposite_dir_members": opposite,
            "queries": [q for _q, q, _b, _c in
                        sorted(pairs, key=lambda x: x[3]["rank"])],
        })
    detail.sort(key=lambda d: (-d["min_gap"],))
    return cls, detail


def sweep(queries_raw, pred, mode, gap_ratios, max_shifts, keep=None):
    by_scene = {}
    for qid, v in queries_raw.items():
        if pred.get(qid):
            by_scene.setdefault(v["visible"], []).append(qid)
    rows = []
    for gr in gap_ratios:
        for ms in max_shifts:
            ms_v = None if ms == "none" else float(ms)
            out = dict(pred)
            ds = []
            for _scene, qids in by_scene.items():
                fixed = enforce_ordinal_chains(
                    [queries_raw[q]["query"] for q in qids],
                    [list(pred[q]) for q in qids],
                    mode=mode, gap_ratio=gr, max_shift=ms_v)
                for q, b1 in zip(qids, fixed):
                    b0 = pred[q]
                    if b1 is not None:
                        out[q] = b1
                        d = max(abs(((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2),
                                abs(((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2))
                        if d > 1e-9:
                            ds.append(round(d, 6))
            st = chain_stats(build_chain_groups_v2(queries_raw, out))
            rows.append({
                "mode": mode, "gap_ratio": gr, "max_shift": ms,
                "ordered": st["ordered"], "broken": st["broken"],
                "A": st["A"], "B_tie": st["B_tie"], "B_dup": st["B_dup"],
                "moved": len(ds),
                "p50": quantile(ds, 0.5), "p90": quantile(ds, 0.9),
                "p99": quantile(ds, 0.99), "max": max(ds) if ds else 0.0,
                "sum": round(sum(ds), 3),
                "gt0.15": sum(1 for d in ds if d > 0.15),
                "gt0.30": sum(1 for d in ds if d > 0.30),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--mode", default="project")
    ap.add_argument("--gap-ratios", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    ap.add_argument("--max-shifts", nargs="+", default=["none", "0.1", "0.2"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(os.path.join(os.path.abspath(args.root), "queries", "queries.json"),
              encoding="utf-8") as fh:
        queries_raw = json.load(fh)
    pred = load_predictions(args.predictions)

    cls, detail = chain_geometry(queries_raw, pred)
    print("=== 破序链几何画像（v2 口径）===")
    for k, v in sorted(cls.items(), key=lambda x: str(x[0])):
        print(f"  {k}: {v}")
    print("\n=== 破序链里 min_gap 最大的 10 条（重指派代价最大）===")
    for d in detail[:10]:
        print(f"  {d['scene']} {d['label']:9s} {d['shape']:10s} n={d['members']}"
              f" span={d['span']:.3f} min_gap={d['min_gap']:.3f}"
              f" opp={d['opposite_dir_members']}")
    print("\n=== 破序链里 min_gap 最小的 5 条（几乎并列）===")
    for d in detail[-5:]:
        print(f"  {d['scene']} {d['label']:9s} {d['shape']:10s} n={d['members']}"
              f" span={d['span']:.3f} min_gap={d['min_gap']:.3f}")

    rows = sweep(queries_raw, pred, args.mode, args.gap_ratios, args.max_shifts)
    print(f"\n=== {args.mode} 模式网格（位移分位数）===")
    print(f"  {'gap':>5s} {'maxshift':>9s} {'ordered':>8s} {'broken':>7s}"
          f" {'A':>4s} {'Btie':>5s} {'moved':>6s} {'p50':>8s} {'p90':>8s}"
          f" {'p99':>8s} {'max':>8s} {'>0.15':>6s} {'>0.30':>6s}")
    for r in rows:
        print(f"  {r['gap_ratio']:>5} {r['max_shift']:>9} {r['ordered']:>8} {r['broken']:>7}"
              f" {r['A']:>4} {r['B_tie']:>5} {r['moved']:>6} {str(r['p50']):>8}"
              f" {str(r['p90']):>8} {str(r['p99']):>8} {r['max']:>8.4f}"
              f" {r['gt0.15']:>6} {r['gt0.30']:>6}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"geometry_classes": {str(k): v for k, v in cls.items()},
                       "broken_detail": detail, "sweep": rows},
                      fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
