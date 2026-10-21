# -*- coding: utf-8 -*-
"""离线：把「共享同一个框的 query」分成可判定两类，量化解歧的收益上限。

问题：`diag_coverage.py` 报 363 条 query 所在图「不同框数 < query 数」，
但其中大量是**正确共指**（多条 query 确实指同一个物体），不能当错误去修。

判据（不需要真值，只依赖 query 自身文本与框几何）：
  - 若一组共享同一框的 query 里，**多条带空间/序数约束**（first/second from left 等）
    且按这些约束**无法指向同一个位置**，则该框至多满足其中一条 —— 其余是**可判定冲突**，
    这就是解歧能救回来的上限。
  - 若一组里最多只有 1 条带约束，或都指向同一侧同一名次，视为**可能正确共指**，不动。

用法：
  python scripts/复赛/dedup_headroom.py \
      --archive outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \
      --json outputs/复赛/diag/dedup_headroom.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.matcher import parse_spatial_constraint  # noqa: E402

PLACEHOLDER = (0.49, 0.49, 0.51, 0.51)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUERIES = os.path.join(PROJECT_ROOT, "dataset", "queries", "queries.json")


def is_ph(b, tol=1e-6):
    return b is not None and all(abs(float(b[i]) - PLACEHOLDER[i]) <= tol for i in range(4))


def load_json_from_any(path):
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            cands = [m for m in t.getmembers()
                     if m.name.endswith("predictions.json") and m.isfile()]
            if not cands:
                cands = [m for m in t.getmembers()
                         if m.name.endswith("submission.json") and m.isfile()]
            with t.extractfile(cands[0]) as fh:
                return json.load(fh)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_queries():
    with open(QUERIES, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "queries" in raw:
        raw = raw["queries"]
    rows = []
    if isinstance(raw, list):
        for rec in raw:
            rows.append((rec["query_id"], rec.get("visible") or rec.get("image"),
                         rec.get("query", "")))
    else:
        for qid, rec in raw.items():
            if isinstance(rec, str):
                rows.append((qid, None, rec))
            else:
                rows.append((qid, rec.get("visible") or rec.get("image"),
                             rec.get("query", "")))
    return rows


def constraint_conflict(cons_list):
    """一组约束能否落在同一个位置上。

    两条约束不冲突的充分条件：axis 与 direction 都相同、且 rank 相差 <= 1
    （相邻名次在 ACC@0.5 下常可互换）；kind=depth 单独处理（与 x/y 轴不直接矛盾）。
    """
    if len(cons_list) < 2:
        return False
    keys = [(c["axis"], c["direction"]) for c in cons_list]
    # 出现方向相反的约束 -> 最不可能同框
    if len(set(keys)) > 1:
        # 深度类约束（closest/farthest）与位置约束可以共存，单独放行
        pos = [(c["axis"], c["direction"], c["rank"]) for c in cons_list
               if c.get("kind") != "depth"]
        if len(pos) >= 2 and len({(a, d) for a, d, _ in pos}) > 1:
            return True
        if len(pos) >= 2:
            ranks = [r for _, _, r in pos]
            return max(ranks) - min(ranks) > 1
        return False
    pos = [(c["axis"], c["rank"]) for c in cons_list if c.get("kind") != "depth"]
    if len(pos) >= 2:
        ranks = [r for _, r in pos]
        return max(ranks) - min(ranks) > 1
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pred = load_json_from_any(args.archive)
    if isinstance(pred, dict) and "results" in pred and isinstance(pred["results"], list):
        pred = {r["query_id"]: r.get("bbox") for r in pred["results"]}

    def box_of(qid):
        v = pred.get(qid)
        if isinstance(v, dict):
            v = v.get("bbox") or v.get("bbox_2d")
        return v

    rows = load_queries()
    by_img = defaultdict(list)
    for qid, vis, q in rows:
        by_img[vis or "?"].append((qid, q))

    tot_shared_groups = tot_shared_q = 0
    conflict_groups = conflict_q = 0
    examples = []
    hist = defaultdict(int)
    gap_images = 0
    gap_q = 0

    for vis, items in by_img.items():
        items.sort(key=lambda x: x[0])
        # 该图真实框按坐标聚类
        groups = defaultdict(list)
        for qid, q in items:
            b = box_of(qid)
            if b is None or is_ph(b):
                continue
            groups[tuple(round(float(x), 6) for x in b)].append((qid, q))

        n_uniq = len(groups)
        if n_uniq < len(items):
            gap_images += 1
            gap_q += len(items) - n_uniq

        for key, members in groups.items():
            if len(members) < 2:
                continue
            tot_shared_groups += 1
            tot_shared_q += len(members) - 1
            hist[len(members)] += 1
            cons = [parse_spatial_constraint(q) for _, q in members]
            cons = [c for c in cons if c]
            if constraint_conflict(cons):
                conflict_groups += 1
                conflict_q += len(members) - 1
                if len(examples) < 20:
                    examples.append({
                        "visible": vis,
                        "n_members": len(members),
                        "constraints": cons,
                        "queries": [q for _, q in members],
                    })

    n_q = sum(len(v) for v in by_img.values())
    print(f"总 query {n_q}  图像 {len(by_img)}")
    print(f"「不同框数 < query 数」的图: {gap_images}  gap 合计 {gap_q} 条 query")
    print(f"共享同一框的组: {tot_shared_groups}  涉及 query（去重后多余条数）: {tot_shared_q}")
    print(f"  └ 其中【约束互相冲突、不可能同框】的组: {conflict_groups}"
          f"  可判定冲突条数: {conflict_q}")
    print(f"多重性直方图 {dict(sorted(hist.items()))}")
    print(f"\n解歧收益上限（可判定冲突条数）= {conflict_q} "
          f"= 全部 query 的 {conflict_q / n_q * 100:.2f}%")

    print("\n冲突组示例：")
    for ex in examples[:10]:
        print(f"  {ex['visible'].split('/')[-1]}  n={ex['n_members']}  "
              f"约束={[(c['axis'], c['direction'], c['rank']) for c in ex['constraints']]}")
        for q in ex["queries"][:3]:
            print(f"      - {q[:90]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "total_queries": n_q,
                "images": len(by_img),
                "gap_images": gap_images,
                "gap_queries": gap_q,
                "shared_groups": tot_shared_groups,
                "shared_queries": tot_shared_q,
                "conflict_groups": conflict_groups,
                "conflict_queries": conflict_q,
                "multiplicity_hist": dict(sorted(hist.items())),
                "examples": examples,
            }, fh, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
