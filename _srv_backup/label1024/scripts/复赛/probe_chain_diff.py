# -*- coding: utf-8 -*-
"""快速探针：统计 diag 落盘里 pre_chain_box != final_box 的规模与分布。"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

DIAG = sys.argv[1] if len(sys.argv) > 1 else \
    "outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl"
TOL = 1e-6


def main():
    recs = []
    with open(DIAG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    print(f"总记录 {len(recs)}")

    def same(a, b):
        if a is None or b is None:
            return a is None and b is None
        return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= TOL

    moved = []
    for r in recs:
        pre, fin = r.get("pre_chain_box"), r.get("final_box")
        if not same(pre, fin):
            moved.append(r)

    print(f"pre_chain_box != final_box 的 query = {len(moved)}")
    print(f"  其中 pre=None 的 = {sum(1 for r in moved if r.get('pre_chain_box') is None)}")
    print(f"  其中 final=None 的 = {sum(1 for r in moved if r.get('final_box') is None)}")

    by_img = defaultdict(list)
    for r in moved:
        by_img[r["visible"]].append(r)
    print(f"涉及图像 = {len(by_img)}")

    dists = []
    for r in moved:
        pre, fin = r["pre_chain_box"], r["final_box"]
        if pre is None or fin is None:
            continue
        cx0, cy0 = (pre[0] + pre[2]) / 2, (pre[1] + pre[3]) / 2
        cx1, cy1 = (fin[0] + fin[2]) / 2, (fin[1] + fin[3]) / 2
        d = (((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)) ** 0.5
        dists.append(d)
    if dists:
        s = sorted(dists)
        n = len(s)
        print(f"位移 p50={s[n // 2]:.4f} p90={s[min(n - 1, int(n * .9))]:.4f} "
              f"max={s[-1]:.4f}  >0.15 的 = {sum(1 for v in s if v > 0.15)}")

    # 每条 query 是否属于某个链（看它是否与同图其他 query 共享 residual key）
    sys.path.insert(0, ".")
    from src.inference.chains import chain_of
    n_chain = sum(1 for r in moved if chain_of(r["query"]) is not None)
    print(f"落在序数链上的 = {n_chain} / {len(moved)}")

    print("\n位移最大的 12 条：")
    rows = []
    for r in moved:
        pre, fin = r.get("pre_chain_box"), r.get("final_box")
        if pre is None or fin is None:
            continue
        d = (((fin[0] + fin[2]) / 2 - (pre[0] + pre[2]) / 2) ** 2 +
             ((fin[1] + fin[3]) / 2 - (pre[1] + pre[3]) / 2) ** 2) ** 0.5
        rows.append((d, r))
    for d, r in sorted(rows, key=lambda x: -x[0])[:12]:
        print(f"  {d:.3f}  {r['query_id']}  {r['visible'].split('/')[-1]}")
        print(f"         pre={[round(v, 3) for v in r['pre_chain_box']]} "
              f"final={[round(v, 3) for v in r['final_box']]}")
        print(f"         Q: {r['query'][:80]}")


if __name__ == "__main__":
    main()