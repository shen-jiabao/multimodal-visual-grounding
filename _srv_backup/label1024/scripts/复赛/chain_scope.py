# -*- coding: utf-8 -*-
"""拆解链修正的**作用范围**：哪些是「修错」（raw 链本身破序），哪些只是「微调」
（raw 链已单调，仅被 gap_ratio 把并列框推开）。

动机：链修正对 116 条 query 有改动，但并非每条都有确定收益。
  - raw 链破序（A / B_*）→ 至少有一个成员必然错，改动是**有据**的；
  - raw 链已 ordered     → 12 个框全在正确位置，任何改动都是**无据**的赌注
    （只可能来自 gap_ratio 把重合框推开，或 project 的 PAVA 平均）。

据此把「116 条改动」拆成两块，给出各自占比，作为是否启用链修正的 EV 依据。

用法：
    python scripts/复赛/chain_scope.py \
        --diag outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl \
        --json outputs/复赛/chain/chain_scope.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.chains import (  # noqa: E402
    chain_of,
    classify_chain,
    enforce_ordinal_chains,
)

TOL = 1e-6
BROKEN = {"A", "B_dup", "B_tie", "B_broken"}


def norm(b):
    return None if b is None else [round(float(v), 6) for v in b]


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--gap-ratio", type=float, default=0.05)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    by_img = defaultdict(list)
    with open(args.diag, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_img[r["visible"]].append(r)

    status_hist = Counter()
    affected_by_status = defaultdict(Counter)
    # 三种修法下「有改动」的 query 下标
    modes = {
        "sort_gap0": ("sort", 0.0),
        "project_gap0": ("project", 0.0),
        f"project_gap{args.gap_ratio}": ("project", args.gap_ratio),
    }
    affected = {k: set() for k in modes}
    group_status_of = {}          # (visible, i) -> status

    n_chain_groups = 0
    none_groups = 0

    for vis, rs in by_img.items():
        rs.sort(key=lambda r: r["query_id"])
        qs = [r["query"] for r in rs]
        raw = [r.get("pre_chain_box") for r in rs]

        groups = defaultdict(list)
        for i, q in enumerate(qs):
            ch = chain_of(q)
            if ch:
                groups[(ch["axis"], ch["direction"], ch["key"])].append(i)

        real = {k: v for k, v in groups.items()
                if len({chain_of(qs[i])["rank"] for i in v}) >= 2}
        if not real:
            continue

        st_map = {}
        for gkey, idxs in real.items():
            n_chain_groups += 1
            if any(raw[i] is None for i in idxs):
                none_groups += 1
                st = "has_none"
            else:
                items = []
                for i in idxs:
                    ch = chain_of(qs[i])
                    items.append((rs[i]["query_id"], qs[i], raw[i],
                                  {"axis": ch["axis"], "direction": ch["direction"],
                                   "rank": ch["rank"]}))
                st = classify_chain(items)
            status_hist[st] += 1
            for i in idxs:
                st_map[i] = st
                group_status_of[(vis, i)] = st

        for name, (mode, gap) in modes.items():
            fixed = enforce_ordinal_chains(qs, raw, mode=mode, gap_ratio=gap)
            for i in range(len(rs)):
                if i in st_map and not same(norm(raw[i]), norm(fixed[i])):
                    affected[name].add((vis, i))
                    affected_by_status[name][st_map[i]] += 1

    print(f"含真链的图 {sum(1 for _ in by_img)}" if False else
          f"读入 {sum(len(v) for v in by_img.values())} 条 query，"
          f"图 {len(by_img)}")
    print(f"真链组数（rank 数>=2）= {n_chain_groups}；其中含 None 框 {none_groups}")
    print("\n=== raw 链状态分布（改动前）===")
    for st, n in status_hist.most_common():
        tag = "← 必然有成员是错的" if st in BROKEN else ""
        print(f"  {st:14s} {n:4d}   {tag}")

    print("\n=== 各修法改动的 query 数，按 raw 链状态拆分 ===")
    header = f"  {'修法':16s} {'改动':>5}  " + "  ".join(
        f"{st:>10s}" for st in sorted(status_hist, key=lambda s: -status_hist[s]))
    print(header)
    for name in modes:
        row = affected_by_status[name]
        cells = "  ".join(f"{row.get(st, 0):>10d}"
                          for st in sorted(status_hist, key=lambda s: -status_hist[s]))
        print(f"  {name:16s} {len(affected[name]):>5}  {cells}")

    print("\n=== 判读 ===")
    for name in modes:
        row = affected_by_status[name]
        n_broken = sum(v for st, v in row.items() if st in BROKEN)
        n_ordered = sum(v for st, v in row.items() if st == "ordered")
        tot = max(1, len(affected[name]))
        print(f"  {name:16s} 修错 {n_broken:3d} ({100.0 * n_broken / tot:5.1f}%)   "
              f"无据改动 {n_ordered:3d} ({100.0 * n_ordered / tot:5.1f}%)   "
              f"其他 {len(affected[name]) - n_broken - n_ordered}")

    # gap_ratio 单独贡献
    g0, g5 = affected["project_gap0"], affected[f"project_gap{args.gap_ratio}"]
    print(f"\n  gap_ratio={args.gap_ratio} 额外带来的改动 = {len(g5 - g0)} 条"
          f"（project 从 {len(g0)} → {len(g5)}）")

    if args.json:
        out = {
            "chain_groups": n_chain_groups,
            "raw_status": dict(status_hist),
            "affected": {k: len(v) for k, v in affected.items()},
            "affected_by_status": {k: dict(v) for k, v in affected_by_status.items()},
            "gap_extra": len(g5 - g0),
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
