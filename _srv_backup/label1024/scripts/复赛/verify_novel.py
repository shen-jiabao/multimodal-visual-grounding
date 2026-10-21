# -*- coding: utf-8 -*-
"""核查 chain_modes_report.py 的「新坐标(novel center)」指标是否可信。

预期：sort 是置换式重指派，其输出中心应等于链内某个原始框的中心；
除非框被 _shift_box 在 [0,1] 边界裁剪，否则 novel 应为 0。
project 做 PAVA 平均，novel 可以很多。

本脚本把 novel 逐条列出来并标注是否触及边界，以确认 86 / 92 这两个数字的含义。
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.chains import chain_of, enforce_ordinal_chains  # noqa: E402

TOL = 0.01
DIAG = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl"


def center(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def norm(b):
    return None if b is None else [round(float(v), 6) for v in b]


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= 1e-6


def main():
    recs = []
    with open(DIAG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    by_img = defaultdict(list)
    for r in recs:
        by_img[r["visible"]].append(r)

    tot = {"sort": 0, "project": 0}
    clipped = {"sort": 0, "project": 0}
    examples = []
    n_aff = 0

    for vis, rs in by_img.items():
        rs.sort(key=lambda r: r["query_id"])
        qs = [r["query"] for r in rs]
        raw = [r.get("pre_chain_box") for r in rs]
        keys = defaultdict(set)
        for q in qs:
            ch = chain_of(q)
            if ch:
                keys[(ch["axis"], ch["direction"], ch["key"])].add(ch["rank"])
        member = set()
        for i, q in enumerate(qs):
            ch = chain_of(q)
            if ch and len(keys[(ch["axis"], ch["direction"], ch["key"])]) >= 2:
                member.add(i)
        if not member:
            continue

        sort_b = enforce_ordinal_chains(qs, raw, mode="sort", gap_ratio=0.0)
        proj_b = enforce_ordinal_chains(qs, raw, mode="project", gap_ratio=0.05)
        aff = [i for i in range(len(rs))
               if not (same(norm(raw[i]), norm(sort_b[i]))
                       and same(norm(raw[i]), norm(proj_b[i])))]
        if not aff:
            continue
        n_aff += len(aff)
        raw_centers = [center(b) for b in raw if b is not None]

        for tag, bs in (("sort", sort_b), ("project", proj_b)):
            for i in aff:
                if bs[i] is None:
                    continue
                c = center(bs[i])
                novel = all(abs(c[0] - rc[0]) > TOL or abs(c[1] - rc[1]) > TOL
                            for rc in raw_centers)
                if not novel:
                    continue
                tot[tag] += 1
                b = bs[i]
                touch = (abs(b[0]) < 1e-6 or abs(b[1]) < 1e-6
                         or abs(b[2] - 1.0) < 1e-6 or abs(b[3] - 1.0) < 1e-6)
                if touch:
                    clipped[tag] += 1
                if len(examples) < 14:
                    examples.append((tag, os.path.basename(vis), rs[i]["query_id"],
                                     qs[i][:60], norm(raw[i]), norm(b), touch))

    print(f"参与修正的 query = {n_aff}")
    print(f"novel（中心距任何原始框中心 > {TOL}）：sort {tot['sort']}  "
          f"project {tot['project']}")
    print(f"  其中输出框触到图像边界（被 _shift_box 裁剪）：sort {clipped['sort']}  "
          f"project {clipped['project']}")
    print("\n样例：")
    for tag, img, qid, q, r, b, touch in examples:
        print(f"  [{tag}] {img} {qid} touch={touch}")
        print(f"        Q   : {q}")
        print(f"        raw : {r}")
        print(f"        out : {b}")


if __name__ == "__main__":
    main()
