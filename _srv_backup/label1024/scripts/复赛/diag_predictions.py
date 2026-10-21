# -*- coding: utf-8 -*-
"""复赛归档预测的离线诊断（只读，不依赖模型权重）。

predictions.json 的格式是 ``{query_id: [x1,y1,x2,y2]}``（非 submission 结构），
因此需要与 ``dataset/queries/queries.json`` 按 query_id join 才能拿到 visible / query。

用途：在没有官方 GT 的前提下，从**预测结构**里榨取可解释的误差信号，
为"提高推理准确度"的方法清单提供可复核的证据。

用法：
    .venv\\Scripts\\python.exe scripts\\复赛\\diag_predictions.py \\
        [--archive outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz] [--json out.json]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import tarfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PLACEHOLDER = [0.49, 0.49, 0.51, 0.51]


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_predictions(path: str):
    if path.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(path, "r:gz") as t:
            name = next(m.name for m in t.getmembers()
                        if os.path.basename(m.name) == "predictions.json")
            return json.load(t.extractfile(name))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="outputs/复赛/archive/"
                                        "run_20260924_090232_qwen_multiimg.tar.gz")
    ap.add_argument("--queries", default="dataset/queries/queries.json")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pred = load_predictions(args.archive)
    with open(args.queries, encoding="utf-8") as f:
        qraw = json.load(f)

    res: dict = {"n_queries_pred": len(pred), "n_queries_json": len(qraw),
                 "n_none": sum(1 for v in pred.values() if v is None)}

    groups: dict = defaultdict(list)
    boxes: dict = {}
    for k, v in pred.items():
        if v is None:
            continue
        vis = qraw.get(k, {}).get("visible")
        if vis is None:
            continue
        groups[vis].append(k)
        boxes[k] = [float(x) for x in v]

    res["n_images"] = len(groups)
    res["n_boxes"] = len(boxes)
    res["n_placeholder"] = sum(1 for b in boxes.values() if b == PLACEHOLDER)

    # 同图内"多个 query 共用一个（近似）框"= 匹配歧义的直接证据
    share = Counter()
    dup_q = 0
    worst = []
    for vis, qs in groups.items():
        c = Counter(tuple(round(x, 4) for x in boxes[q]) for q in qs)
        for _, n in c.items():
            share[n] += 1
        s = sum(n for n in c.values() if n > 1)
        dup_q += s
        if s:
            top = max(c.items(), key=lambda kv: kv[1])
            worst.append({"visible": vis, "queries": len(qs), "sharing": s,
                          "max_same_box": top[1],
                          "qids": [q for q in qs
                                   if tuple(round(x, 4) for x in boxes[q]) == top[0]][:8]})
    res["exact_box_multiplicity"] = dict(sorted(share.items()))
    res["queries_sharing_exact_box"] = dup_q
    res["share_ratio"] = round(dup_q / max(1, len(boxes)), 4)
    res["images_with_shared_box"] = len(worst)
    res["worst_shared"] = sorted(worst, key=lambda r: -r["max_same_box"])[:6]

    # 近似重复（IoU>=0.9）
    near = tot = 0
    for vis, qs in groups.items():
        qs = qs[:60]
        for a, b in itertools.combinations(qs, 2):
            tot += 1
            if iou(boxes[a], boxes[b]) >= 0.9:
                near += 1
    res["near_dup_pairs_iou90"] = near
    res["pairs_compared"] = tot

    # 框几何
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b in boxes.values())
    n = len(areas)
    res["area"] = {
        "min": areas[0], "p10": areas[int(0.10 * n)], "median": statistics.median(areas),
        "p90": areas[int(0.90 * n)], "max": areas[-1],
        "degenerate_w_or_h_lt_0.01": sum(1 for b in boxes.values()
                                         if (b[2] - b[0]) < 0.01 or (b[3] - b[1]) < 0.01),
        "area_gt_0.5": sum(1 for a in areas if a > 0.5),
    }
    res["clamp"] = {
        "touch_x1y1_zero": sum(1 for b in boxes.values() if b[0] <= 1e-6 or b[1] <= 1e-6),
        "touch_x2y2_one": sum(1 for b in boxes.values() if b[2] >= 0.999999 or b[3] >= 0.999999),
    }
    res["queries_per_image"] = dict(sorted(Counter(len(v) for v in groups.values()).items()))

    # 同图内 query 之间的词面相似度（匹配歧义的风险面大小）
    jac = []
    for vis, qs in groups.items():
        if len(qs) < 2 or len(qs) > 40:
            continue
        for a, b in itertools.combinations(qs, 2):
            wa = set(qraw[a]["query"].lower().split())
            wb = set(qraw[b]["query"].lower().split())
            if wa and wb:
                jac.append(len(wa & wb) / len(wa | wb))
    if jac:
        jac.sort()
        res["within_image_query_jaccard"] = {
            "n_pairs": len(jac), "median": statistics.median(jac),
            "p90": jac[int(0.9 * len(jac))],
            "ge_0.5": sum(1 for x in jac if x >= 0.5),
            "ge_0.75": sum(1 for x in jac if x >= 0.75),
        }

    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("已导出:", args.json)


if __name__ == "__main__":
    main()
