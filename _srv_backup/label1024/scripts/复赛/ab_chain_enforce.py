# -*- coding: utf-8 -*-
"""序数链修正的离线 A/B：在**已有预测**上量测 enforce_ordinal_chains 的效果。

口径（重要，v2）：
    分组键与 ``src.inference.chains.enforce_ordinal_chains`` 内部完全一致 ——
    ``(scene, axis, direction, residual_key)``。旧的 `analyze_chain_projection.py`
    用 ``(scene, residual_key)``（会把方向/轴向冲突的成员混在一组里，整组被判
    ``skip_axis`` 并从 A_ratio 分母里静默丢弃），因此那里的 before/after 与
    函数实际作用范围不一致。本脚本用 v2 口径重做。

判据：
    before/after 都按 v2 口径数 ordered / broken（A / B_dup / B_tie / B_broken）。
    另外报告位移统计与"高风险重指派"计数 —— 排序式重指派（sort）在交换两个
    相距较远的框时，平均位移可达 0.1（图像宽度 10%），在 ACC@0.5 下是有风险的；
    PAVA 投影（project）只做最小 L2 微调、不改变指派，位移显著更小。

用法：
    python scripts/复赛/ab_chain_enforce.py \
        --predictions outputs/复赛/archive/run_..._qwen_multiimg.tar.gz \
        --modes sort project --gap-ratios 0.0 0.1 --max-shifts none 0.05 0.1
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
    measure,
)
from src.inference.chains import enforce_ordinal_chains  # noqa: E402


def apply_enforce(queries_raw, pred, mode, gap_ratio, max_shift):
    """按图施加序数链修正，返回 (新预测, 位移列表)。"""
    by_scene = {}
    for qid, v in queries_raw.items():
        if pred.get(qid):
            by_scene.setdefault(v["visible"], []).append(qid)
    out = dict(pred)
    shifts = []
    for _scene, qids in by_scene.items():
        queries = [queries_raw[q]["query"] for q in qids]
        boxes = [list(pred[q]) for q in qids]
        fixed = enforce_ordinal_chains(queries, boxes, mode=mode,
                                       gap_ratio=gap_ratio, max_shift=max_shift)
        for q, b0, b1 in zip(qids, boxes, fixed):
            if b1 is None:
                continue
            out[q] = b1
            c0 = ((b0[0] + b0[2]) / 2, (b0[1] + b0[3]) / 2)
            c1 = ((b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2)
            d = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))
            if d > 1e-9:
                shifts.append((q, d))
    return out, shifts


def risk_scan(queries_raw, pred):
    """量化 sort 式重指派的风险：按"框中心是否被换到别的框位置"分类破序链。"""
    groups = build_chain_groups_v2(queries_raw, pred)
    cat = Counter()
    swap_dists = []
    for _key, items in groups.items():
        if len(items) < 2:
            continue
        from src.inference.chains import classify_chain
        label = classify_chain(items)
        if label in ("ordered", "skip_axis", "skip_rank"):
            continue
        cat[label] += 1
        after, _ = apply_enforce(
            {items[0][0]: queries_raw[items[0][0]]}, pred, "sort", 0.0, None)
        # 链内两两中心距离：用于判断"框是否相距很远（重指派代价大）"
        cons = items[0][3]
        axis_i = 0 if cons["axis"] == "x" else 1
        cs = sorted((b[0] + b[2]) / 2 if axis_i == 0 else (b[1] + b[3]) / 2
                    for _, _, b, _ in items)
        for i in range(len(cs) - 1):
            swap_dists.append(cs[i + 1] - cs[i])
    swap_dists.sort()
    return {
        "broken_by_label": dict(cat),
        "broken_total": sum(cat.values()),
        "adjacent_gap_median": round(swap_dists[len(swap_dists) // 2], 5) if swap_dists else 0.0,
        "adjacent_gap_p10": round(swap_dists[max(0, len(swap_dists) // 10)], 5) if swap_dists else 0.0,
        "adjacent_gap_gt0.3": sum(1 for d in swap_dists if d > 0.3),
        "adjacent_gap_le0.05": sum(1 for d in swap_dists if d <= 0.05),
        "adjacent_pairs": len(swap_dists),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--modes", nargs="+", default=["sort", "project"])
    ap.add_argument("--gap-ratios", nargs="+", type=float, default=[0.0])
    ap.add_argument("--max-shifts", nargs="+", default=["none"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    with open(os.path.join(os.path.abspath(args.root), "queries", "queries.json"),
              encoding="utf-8") as fh:
        queries_raw = json.load(fh)
    pred = load_predictions(args.predictions)

    base = measure(pred, queries_raw)
    print("BEFORE(ordinal v2): " + json.dumps(base["ordinal"], ensure_ascii=False))
    print("风险扫描: " + json.dumps(risk_scan(queries_raw, pred), ensure_ascii=False))

    rows = []
    for mode in args.modes:
        for gr in args.gap_ratios:
            for ms in args.max_shifts:
                max_shift = None if ms == "none" else float(ms)
                out, shifts = apply_enforce(queries_raw, pred, mode, gr, max_shift)
                st = chain_stats(build_chain_groups_v2(queries_raw, out))
                m = measure(out, queries_raw)
                ds = [d for _, d in shifts]
                row = {
                    "mode": mode, "gap_ratio": gr, "max_shift": ms,
                    "ordered": st["ordered"], "broken": st["broken"],
                    "A": st["A"], "B_dup": st["B_dup"], "B_tie": st["B_tie"],
                    "B_broken": st["B_broken"],
                    "A_ratio": st["A_ratio"], "ordered_ratio": st["ordered_ratio"],
                    "boxes_moved": len(ds),
                    "mean_shift": round(sum(ds) / len(ds), 5) if ds else 0.0,
                    "max_shift_obs": round(max(ds), 5) if ds else 0.0,
                    "placeholder": m["coverage"]["placeholder"],
                    "shared_box": m["collapse"]["queries_in_shared_box"],
                    "iou90": m["collapse"]["pairs_iou_ge90"],
                    "touch_x1y1": m["geometry"]["touch_x1y1_zero"],
                    "touch_x2y2": m["geometry"]["touch_x2y2_one"],
                    "degenerate": m["coverage"]["degenerate"],
                }
                rows.append(row)
                print("\n--- mode=%s gap_ratio=%s max_shift=%s ---" % (mode, gr, ms))
                print("  ordered %d->%d   broken %d->%d (A %d, B_dup %d, B_tie %d, B_broken %d)"
                      % (base["ordinal"]["ordered"], st["ordered"],
                         base["ordinal"]["broken"], st["broken"],
                         st["A"], st["B_dup"], st["B_tie"], st["B_broken"]))
                print("  A_ratio %s -> %s   ordered_ratio %s -> %s"
                      % (base["ordinal"]["A_ratio"], st["A_ratio"],
                         base["ordinal"]["ordered_ratio"], st["ordered_ratio"]))
                print("  移动框 %d  平均位移 %.4f  最大位移 %.4f"
                      % (len(ds), row["mean_shift"], row["max_shift_obs"]))
                print("  副作用: 占位 %d->%d  共框 %d->%d  IoU90对 %d->%d"
                      "  贴左边 %d->%d  贴右边 %d->%d"
                      % (base["coverage"]["placeholder"], m["coverage"]["placeholder"],
                         base["collapse"]["queries_in_shared_box"],
                         m["collapse"]["queries_in_shared_box"],
                         base["collapse"]["pairs_iou_ge90"], m["collapse"]["pairs_iou_ge90"],
                         base["geometry"]["touch_x1y1_zero"], m["geometry"]["touch_x1y1_zero"],
                         base["geometry"]["touch_x2y2_one"], m["geometry"]["touch_x2y2_one"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"before": base["ordinal"], "risk": risk_scan(queries_raw, pred),
                       "rows": rows}, fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
