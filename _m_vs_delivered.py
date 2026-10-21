#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M 件 vs 交付件（0.7963）的离线对表：改了哪些、改多少、改在哪一层。

不看分数，只看几何差异；并按"序数 1103 / 非序数"分层（序数集取自
`_ordinal_cores_r2.json` 的键）。用途：判断 M 是否值得占用一个提交名额。
"""
from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "_m_direct.json")
REF = os.path.join(HERE, "submission_zips", "semifinal", "0.7963",
                   "result_semi_q38_nonord.json")
CORES = os.path.join(HERE, "_ordinal_cores_r2.json")


def box(v):
    if isinstance(v, dict):
        v = v.get("bbox")
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except Exception:
            return None
    return None


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    m = json.load(open(M, encoding="utf-8"))
    r = json.load(open(REF, encoding="utf-8"))
    cores = json.load(open(CORES, encoding="utf-8"))
    ordinal = set(cores)

    print(f"M 条目 {len(m)}｜交付件 {len(r)}｜序数集 {len(ordinal)}")

    stats = {"序数": {"同": 0, "改": 0, "无框": 0, "iou": []},
             "非序数": {"同": 0, "改": 0, "无框": 0, "iou": []}}
    layers = {"[0,0.5) 换实例": 0, "[0.5,0.9) 框偏移": 0,
              "[0.9,0.999) 框微偏": 0, "[0.999,1] 几乎不动": 0}
    for k, v in m.items():
        grp = "序数" if k in ordinal else "非序数"
        mb = box(v)
        rb = box(r.get(k))
        if mb is None:
            stats[grp]["无框"] += 1
            continue
        if rb is None:
            stats[grp]["改"] += 1
            continue
        j = iou(mb, rb)
        stats[grp]["iou"].append(j)
        if j >= 0.999:
            stats[grp]["同"] += 1
            layers["[0.999,1] 几乎不动"] += 1
        else:
            stats[grp]["改"] += 1
            if j >= 0.9:
                layers["[0.9,0.999) 框微偏"] += 1
            elif j >= 0.5:
                layers["[0.5,0.9) 框偏移"] += 1
            else:
                layers["[0,0.5) 换实例"] += 1

    print()
    print("=== 分层结果（M vs 交付件）===")
    for grp in ("序数", "非序数"):
        s = stats[grp]
        L = s["iou"]
        med = st.median(L) if L else float("nan")
        print(f"  {grp}: 同(IoU≥0.999) {s['同']}｜改 {s['改']}"
              f"｜M 无框(留基线) {s['无框']}｜可比 {len(L)}"
              f"｜中位 IoU {med:.4f}")

    print()
    print("=== 改动条的几何分层（全部）===")
    tot = sum(layers.values())
    for k, c in layers.items():
        print(f"  {k:<20} {c:>5}  ({c/max(1,tot)*100:5.1f}%)")

    print()
    print("=== 关键判断量 ===")
    n_ord = stats["序数"]
    n_non = stats["非序数"]
    print(f"  序数层改动 {n_ord['改']} / 1103（{n_ord['改']/1103*100:.1f}%）")
    print(f"  非序数层改动 {n_non['改']} / {len(m)-1103}"
          f"（{n_non['改']/max(1,len(m)-1103)*100:.1f}%）")
    print(f"  合计改动 {n_ord['改']+n_non['改']}")
    print(f"  M 无框留基线合计 {n_ord['无框']+n_non['无框']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
