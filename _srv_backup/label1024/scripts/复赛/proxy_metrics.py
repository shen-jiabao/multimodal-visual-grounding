# -*- coding: utf-8 -*-
"""统一代理指标测量（复赛，无 GT 下的改动前后标尺）。

设计目的（对应用户 m00170「执行」第 0 阶段）：
    无真值环境下，任何改动都必须先证明它减少的是"确定性错误"，
    而不是把正确框改坏。本脚本把这类可离线量测的指标收敛成**一条命令**：

        python scripts/复赛/proxy_metrics.py --predictions A.json            # 基线表
        python scripts/复赛/proxy_metrics.py --predictions A.json \
               --compare B.json --json out.json                              # 前后 diff 表

指标分五组：
  1) 完整性 coverage        ：覆盖条数、缺失、占位框（必错的 [.49,.49,.51,.51]）、退化框
  2) 歧义性 collapse        ：同图共用同框的条数 / 多重性直方图 / 近重复对 IoU 分布
  3) 几何合法性 geometry    ：越界、反向、贴边（x1/y1≈0、x2/y2≈1）、面积分布
  4) 序数链 ordinal         ：ordered/A/B_dup/B_tie/B_broken 与 A_ratio
  5) 能力面 capability      ：按 query 语义标签的分布（spatial/part/ordinal/depth...）

另提供 `--oracle-chain`：算出"若把每条破序链按位置重指派，需要移动多少框、
多大位移"，即 chain_enforce 的收益上限（离线即可知，无需上云）。

坐标约定：predictions 为 {query_id: [x1, y1, x2, y2]}，域 [0,1]。
本脚本只读，不修改任何结果文件；除非显式传 --dump-* 才会写出派生文件。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.chains import (  # noqa: E402
    classify_chain,
    enforce_ordinal_chains,
    residual_tokens,
)
from src.inference.matcher import parse_spatial_constraint  # noqa: E402

# 配置默认占位框（configs/config.yaml: fallback_bbox）
PLACEHOLDER = (0.49, 0.49, 0.51, 0.51)
PLACEHOLDER_TOL = 1e-6

# ---------------------------------------------------------------- query 标签
_LABELS = {
    "color": r"\b(red|blue|green|yellow|orange|purple|pink|black|white|gray|grey|brown|silver|golden|blond|dark|light)\w*\b",
    "ordinal": r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|leftmost|rightmost|topmost|bottommost|next to last)\b",
    "spatial": r"\b(left|right|top|bottom|above|below|beside|behind|front|near|next to|between|under|over|inside|outside|across|along)\b",
    "depth": r"\b(closest|closer|nearest|farthest|furthest|farther|deep|depth|distance|away)\b",
    "relational": r"\b(holding|holds|wearing|wears|carrying|carries|riding|rides|standing|sitting|walking|talking|looking|facing|beside|with|next to|on the|under the|in the)\b",
    "part": r"\b(head|hand|hands|arm|arms|leg|legs|foot|feet|face|eye|eyes|ear|nose|mouth|hair|wheel|wheels|handle|seat|screen|door|window|tail|wing|neck|back|shoulder|knee|finger|pocket|sleeve|collar)\b",
    "action": r"\b(running|walking|jumping|sitting|standing|lying|holding|riding|eating|drinking|playing|talking|smiling|laughing|reading|writing|throwing|carrying)\b",
    "superlative": r"\b(largest|smallest|biggest|tallest|shortest|longest|highest|lowest|widest|narrowest|closest|farthest|most|least)\b",
    "thermal": r"\b(hot|cold|warm|cool|heat|thermal|temperature|infrared)\b",
}
_LABELS_C = {k: re.compile(v) for k, v in _LABELS.items()}
_CJK = re.compile(r"[\u4e00-\u9fff]")


def capability_tags(query: str) -> list:
    q = query.lower()
    tags = [k for k, rx in _LABELS_C.items() if rx.search(q)]
    if _CJK.search(query):
        tags.append("cjk")
    return tags


# ---------------------------------------------------------------- 基础几何
def load_predictions(path):
    """读取预测结果；支持三种输入形态：

    1. 直接的 {query_id: [x1,y1,x2,y2]} json；
    2. 归档运行目录里的 predictions.json；
    3. 归档 tar.gz（run_*.tar.gz，内部含 predictions.json）。

    背景：本机沙箱对 outputs/复赛/archive/run_* 目录本身拒绝列目录/读取，
    但同目录的 .tar.gz 可读（读工具也可读目录内单文件），故自动回退到压缩包。
    """
    if path.endswith(".tar.gz") or path.endswith(".tgz"):
        import tarfile
        with tarfile.open(path, "r:gz") as tf:
            name = next((m.name for m in tf.getmembers()
                         if m.name.endswith("predictions.json")), None)
            if name is None:
                raise SystemExit(f"[error] 压缩包内找不到 predictions.json: {path}")
            with tf.extractfile(name) as fh:
                return json.loads(fh.read().decode("utf-8"))
    if os.path.isdir(path):
        cand = os.path.join(path, "predictions.json")
        if os.path.exists(cand):
            path = cand
        else:
            tgz = [f for f in os.listdir(path) if f.endswith(".tar.gz")]
            if tgz:
                return load_predictions(os.path.join(path, tgz[0]))
            raise SystemExit(f"[error] 目录内找不到 predictions.json: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def is_placeholder(b) -> bool:
    return all(abs(float(b[i]) - PLACEHOLDER[i]) <= PLACEHOLDER_TOL for i in range(4))


def is_degenerate(b) -> bool:
    if any(v != v for v in b):  # NaN
        return True
    return (b[2] - b[0]) <= 1e-6 or (b[3] - b[1]) <= 1e-6 or b[2] < b[0] or b[3] < b[1]


def pct(n, d) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def quantile(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return round(s[i], 6)


# ---------------------------------------------------------------- 链分组
def build_chain_groups(queries_raw, pred):
    """按 (visible, residual_tokens) 归链。

    .. warning::
       这是 ``scripts/analyze_ordinal_chains.py`` 的**历史口径**，与
       :func:`src.inference.chains.enforce_ordinal_chains` 实际使用的
       (轴向, 方向, 残余词) 分组键**不一致**：同一残余词集合里若混入
       "from left to right" 与 "from right to left"，整组会被
       :func:`classify_chain` 判为 ``skip_axis``，从而在 A_ratio 的分母里
       被静默丢弃。历史归档里那个 A_ratio=0.7027 就是在这个口径下算出的，
       分母只有 37 条链，而实际可用的有序链基准远多于这个数。
       新代码请用 :func:`build_chain_groups_v2`。
    """
    groups = {}
    for qid, v in queries_raw.items():
        b = pred.get(qid)
        if not b:
            continue
        cons = parse_spatial_constraint(v["query"])
        if cons is None or cons.get("kind") == "depth" or int(cons.get("rank", 0)) <= 0:
            continue
        groups.setdefault((v["visible"], residual_tokens(v["query"])), []).append(
            (qid, v["query"], list(b), cons))
    return groups


def build_chain_groups_v2(queries_raw, pred):
    """正确口径：按 (scene, axis, direction, key) 归链。

    与 :func:`src.inference.chains.enforce_ordinal_chains` 内部的分组键
    ``(axis, direction, key)`` 完全对齐，因此按此口径测量 before/after
    才真正反映该函数的作用范围。
    """
    from src.inference.chains import chain_of
    groups = {}
    for qid, v in queries_raw.items():
        b = pred.get(qid)
        if not b:
            continue
        ch = chain_of(v["query"])
        if ch is None:
            continue
        cons = parse_spatial_constraint(v["query"])
        groups.setdefault(
            (v["visible"], ch["axis"], ch["direction"], ch["key"]), []
        ).append((qid, v["query"], list(b), cons))
    return groups


def chain_stats(groups):
    c = Counter()
    per_axis = Counter()
    for _key, items in groups.items():
        if len(items) < 2:
            continue
        label = classify_chain(items)
        c[label] += 1
        if not label.startswith("skip"):
            per_axis[items[0][3]["axis"]] += 1
    # 分母 = 参与判定的链 = 有序 + 全部破序（skip 组不参与）
    denom = c["ordered"] + c["A"] + c["B_dup"] + c["B_tie"] + c["B_broken"]
    broken = c["A"] + c["B_dup"] + c["B_tie"] + c["B_broken"]
    return {
        "groups": sum(c.values()),
        "skip_axis": c["skip_axis"],
        "skip_rank": c["skip_rank"],
        "valid": denom,
        "ordered": c["ordered"],
        "broken": broken,
        "A": c["A"],
        "B_dup": c["B_dup"],
        "B_tie": c["B_tie"],
        "B_broken": c["B_broken"],
        "A_ratio": round(c["A"] / denom, 4) if denom else None,
        "ordered_ratio": round(c["ordered"] / denom, 4) if denom else None,
        "axis_x": per_axis["x"],
        "axis_y": per_axis["y"],
    }


def oracle_chain_move(queries_raw, pred):
    """收益上限：把每条破序链按位置重指派，需要移动多少框、平均/最大位移多大。

    输入口径 v2（按轴/方向/残余词归链），与 enforce_ordinal_chains 作用范围一致。
    只统计**非 skip 的破序链**（skip = 组内 rank 不连续或轴/方向不一致，
    函数本身也会跳过，不产生位移）。
    """
    groups = build_chain_groups_v2(queries_raw, pred)
    moved, shifts, upper, bad_groups = 0, [], 0.0, 0
    for _key, items in groups.items():
        if len(items) < 2:
            continue
        label = classify_chain(items)
        if label in ("ordered", "skip_axis", "skip_rank"):
            continue
        bad_groups += 1
        cons = items[0][3]
        axis_i = 0 if cons["axis"] == "x" else 1
        after = enforce_ordinal_chains(
            [q for _, q, _, _ in items], [b for _, _, b, _ in items], mode="sort")
        for (_, _, b0, _), b1 in zip(items, after):
            c0 = (b0[0] + b0[2]) / 2 if axis_i == 0 else (b0[1] + b0[3]) / 2
            c1 = (b1[0] + b1[2]) / 2 if axis_i == 0 else (b1[1] + b1[3]) / 2
            upper += abs(c1 - c0)
            if abs(c1 - c0) > 1e-9:
                moved += 1
                shifts.append(abs(c1 - c0))
    n = len(shifts)
    return {
        "broken_groups": bad_groups,
        "boxes_moved": moved,
        "mean_shift": round(sum(shifts) / n, 5) if n else 0.0,
        "max_shift": round(max(shifts), 5) if n else 0.0,
        "sum_shift": round(upper, 3),
    }


# ---------------------------------------------------------------- 主测量
def measure(pred, queries_raw, depth_stats=None):
    out = {}
    all_ids = list(queries_raw.keys())
    n = len(all_ids)

    # 1) 完整性
    missing = [q for q in all_ids if q not in pred]
    none_cnt = sum(1 for q in all_ids if pred.get(q) is None)
    present = [q for q in all_ids if pred.get(q)]
    ph = sum(1 for q in present if is_placeholder(pred[q]))
    degen = sum(1 for q in present if is_degenerate(pred[q]))
    out["coverage"] = {
        "queries": n,
        "predicted": len(present),
        "missing": len(missing),
        "none": none_cnt,
        "placeholder": ph,
        "placeholder_pct": pct(ph, len(present)),
        "degenerate": degen,
        "images": len({v["visible"] for v in queries_raw.values()}),
    }

    # 按图归组
    by_img = {}
    for q in present:
        by_img.setdefault(queries_raw[q]["visible"], []).append(q)

    # 2) 歧义性
    mult = Counter()
    pair_iou = []
    same_img_queries = 0
    for vis, qs in by_img.items():
        same_img_queries += len(qs)
        boxes = {q: tuple(pred[q]) for q in qs}
        cnt = Counter(boxes.values())
        mult.update(cnt.values())
        uniq = list(cnt.keys())
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_iou.append(iou(uniq[i], uniq[j]))
    # 共用同框的 query 条数 = Σ (多重性-1) × 该多重性的框数
    shared_q = sum((m - 1) * c for m, c in mult.items() if m >= 2)
    out["collapse"] = {
        "distinct_boxes": sum(mult.values()),
        "queries_in_shared_box": shared_q,
        "queries_in_shared_box_pct": pct(shared_q, len(present)),
        "multiplicity_hist": {int(m): int(c) for m, c in sorted(mult.items())},
        "boxes_mult_ge4": sum(c for m, c in mult.items() if m >= 4),
        "max_multiplicity": max(mult) if mult else 0,
        "pair_compared": len(pair_iou),
        "pairs_iou_ge70": sum(1 for v in pair_iou if v >= 0.7),
        "pairs_iou_ge90": sum(1 for v in pair_iou if v >= 0.9),
    }

    # 3) 几何合法性
    left_top = right_bottom = oob = rev = 0
    areas, ws, hs = [], [], []
    for q in present:
        b = pred[q]
        if any(v != v for v in b):
            continue
        if b[0] <= 1e-6 or b[1] <= 1e-6:
            left_top += 1
        if b[2] >= 1 - 1e-6 or b[3] >= 1 - 1e-6:
            right_bottom += 1
        if b[0] < -1e-6 or b[1] < -1e-6 or b[2] > 1 + 1e-6 or b[3] > 1 + 1e-6:
            oob += 1
        if b[2] < b[0] or b[3] < b[1]:
            rev += 1
        areas.append((b[2] - b[0]) * (b[3] - b[1]))
        ws.append(b[2] - b[0])
        hs.append(b[3] - b[1])
    out["geometry"] = {
        "touch_x1y1_zero": left_top,
        "touch_x2y2_one": right_bottom,
        "out_of_unit": oob,
        "reversed": rev,
        "w_lt_0.01": sum(1 for v in ws if v < 0.01),
        "h_lt_0.01": sum(1 for v in hs if v < 0.01),
        "area_gt_0.5": sum(1 for v in areas if v > 0.5),
        "area_p10": quantile(areas, 0.10),
        "area_median": quantile(areas, 0.50),
        "area_p90": quantile(areas, 0.90),
    }

    # 4) 序数链 —— v2 为准（与 enforce_ordinal_chains 分组键一致）
    out["ordinal"] = chain_stats(build_chain_groups_v2(queries_raw, pred))
    out["ordinal_legacy"] = chain_stats(build_chain_groups(queries_raw, pred))

    # 5) 能力面
    tag = Counter()
    tag_queries = Counter()
    for v in queries_raw.values():
        ts = capability_tags(v["query"])
        for t in ts:
            tag[t] += 1
        tag_queries[len(ts)] += 1
    words = [len(v["query"].split()) for v in queries_raw.values()]
    out["capability"] = {
        "by_tag": {k: {"n": tag[k], "pct": pct(tag[k], n)} for k in sorted(tag)},
        "tags_per_query_hist": {int(k): int(v) for k, v in sorted(tag_queries.items())},
        "words_median": quantile(words, 0.5),
        "words_p90": quantile(words, 0.9),
        "words_ge20": sum(1 for w in words if w >= 20),
        "query_pct_none": 0.0 if not present else round(100.0 * ph / len(present), 2),
    }
    if depth_stats:
        out["depth"] = depth_stats
    return out


# ---------------------------------------------------------------- 深度一致性
def depth_consistency(pred, queries_raw, depth_dir):
    """深度语义 query 的跨模态一致性：框内平均深度序 与 框中心纵/横坐标序 的一致率。

    说明：这不是准确率，只是"深度模态是否被真正利用"的证据。
    深度先验要能改变结果，前提是模型给出的框按深度排序与按几何排序不一致。
    """
    import numpy as np
    from PIL import Image

    ORDER_RE = re.compile(r"\b(closest|nearest|farthest|furthest|closer|farther)\b")
    groups = {}
    for qid, v in queries_raw.items():
        if not pred.get(qid):
            continue
        if not ORDER_RE.search(v["query"].lower()):
            continue
        groups.setdefault((v["visible"], residual_tokens(v["query"])), []).append(
            (qid, v["query"], pred[qid]))
    res = {"groups": 0, "pairs": 0, "agree": 0, "mean_abs_rank_diff": 0.0,
           "groups_with_depth": 0}
    diffs = []
    for (vis, _key), items in groups.items():
        if len(items) < 2:
            continue
        res["groups"] += 1
        path = os.path.join(depth_dir, os.path.basename(vis))
        if not os.path.exists(path):
            continue
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        H, W = arr.shape
        vals = []
        for _qid, _q, b in items:
            x1 = int(max(0, min(W - 1, round(b[0] * W))))
            x2 = int(max(0, min(W, round(b[2] * W))))
            y1 = int(max(0, min(H - 1, round(b[1] * H))))
            y2 = int(max(0, min(H, round(b[3] * H))))
            if x2 <= x1 or y2 <= y1:
                vals.append(None)
            else:
                vals.append(float(arr[y1:y2, x1:x2].mean()))
        if any(v is None for v in vals):
            continue
        res["groups_with_depth"] += 1
        rank_d = sorted(range(len(vals)), key=lambda i: vals[i])
        rank_g = sorted(range(len(items)),
                        key=lambda i: (items[i][2][1] + items[i][2][3]) / 2)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                res["pairs"] += 1
                a = rank_d.index(i) - rank_d.index(j)
                b = rank_g.index(i) - rank_g.index(j)
                if a * b > 0:
                    res["agree"] += 1
                diffs.append(abs(rank_d.index(i) - rank_g.index(i)))
    res["agree_ratio"] = round(res["agree"] / res["pairs"], 4) if res["pairs"] else None
    res["mean_abs_rank_diff"] = round(sum(diffs) / len(diffs), 3) if diffs else 0.0
    return res


# ---------------------------------------------------------------- 输出
SCALARS = [
    ("coverage", "predicted"), ("coverage", "missing"), ("coverage", "none"),
    ("coverage", "placeholder"), ("coverage", "degenerate"),
    ("collapse", "queries_in_shared_box"), ("collapse", "pairs_iou_ge90"),
    ("collapse", "boxes_mult_ge4"), ("collapse", "max_multiplicity"),
    ("geometry", "touch_x1y1_zero"), ("geometry", "touch_x2y2_one"),
    ("geometry", "out_of_unit"), ("geometry", "reversed"),
    ("geometry", "w_lt_0.01"), ("geometry", "h_lt_0.01"), ("geometry", "area_gt_0.5"),
    ("ordinal", "valid"), ("ordinal", "ordered"), ("ordinal", "A"),
    ("ordinal", "B_dup"), ("ordinal", "B_tie"), ("ordinal", "B_broken"),
    ("ordinal", "A_ratio"), ("ordinal", "ordered_ratio"),
]


def flat(d):
    return {f"{g}.{k}": d.get(g, {}).get(k) for g, k in SCALARS}


def print_report(name, m, baseline=None):
    print(f"\n===== {name} =====")
    f = flat(m)
    if baseline is None:
        for k, v in f.items():
            print(f"  {k:34s} {v}")
    else:
        b = flat(baseline)
        print(f"  {'指标':34s} {'before':>12s} {'after':>12s} {'delta':>10s}")
        for k in f:
            bv, av = b.get(k), f[k]
            d = ""
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                d = f"{av - bv:+g}"
            print(f"  {k:34s} {str(bv):>12s} {str(av):>12s} {d:>10s}")
    print(f"  multiplicity_hist                {m['collapse']['multiplicity_hist']}")
    print("  ordinal_v2                       " + json.dumps(m["ordinal"], ensure_ascii=False))
    print("  ordinal_legacy(旧口径)            "
          + json.dumps(m.get("ordinal_legacy", {}), ensure_ascii=False))
    cap = m["capability"]["by_tag"]
    print("  capability(n/pct)                " + ", ".join(
        f"{k}:{v['n']}({v['pct']}%)" for k, v in cap.items()))
    if "depth" in m:
        print(f"  depth_consistency                {json.dumps(m['depth'], ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--predictions", required=True, help="A：基线预测 json")
    ap.add_argument("--compare", default=None, help="B：对比预测 json（给出 diff 表）")
    ap.add_argument("--json", default=None, help="导出统计结果")
    ap.add_argument("--oracle-chain", action="store_true",
                    help="额外计算序数链修正的收益上限（需移动框数/位移）")
    ap.add_argument("--depth", action="store_true",
                    help="额外计算深度语义 query 的一致性（需读深度图，较慢）")
    ap.add_argument("--depth-dir", default=None)
    args = ap.parse_args()

    with open(os.path.join(os.path.abspath(args.root), "queries", "queries.json"),
              encoding="utf-8") as fh:
        queries_raw = json.load(fh)
    pred_a = load_predictions(args.predictions)

    depth_dir = args.depth_dir or os.path.join(os.path.abspath(args.root), "Images", "depth")
    if args.depth and not os.path.isdir(depth_dir):
        print(f"[warn] 深度目录不存在，跳过：{depth_dir}")
        args.depth = False

    depth_stats = depth_consistency(pred_a, queries_raw, depth_dir) if args.depth else None
    m_a = measure(pred_a, queries_raw, depth_stats)

    m_b = None
    if args.compare:
        pred_b = load_predictions(args.compare)
        m_b = measure(pred_b, queries_raw, None)

    print_report("A baseline: " + args.predictions, m_a)
    if m_b:
        print_report("B compare: " + args.compare, m_b, baseline=m_a)

    if args.oracle_chain:
        orc = oracle_chain_move(queries_raw, pred_a)
        print("\n----- 序数链修正收益上限（离线，oracle 重指派，v2 口径） -----")
        print("  " + json.dumps(orc, ensure_ascii=False))

    if args.json:
        payload = {"predictions_a": args.predictions,
                   "predictions_b": args.compare,
                   "metrics_a": m_a, "metrics_b": m_b}
        if args.oracle_chain:
            payload["oracle_chain"] = oracle_chain_move(queries_raw, pred_a)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
