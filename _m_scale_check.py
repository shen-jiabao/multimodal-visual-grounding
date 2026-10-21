#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并图坐标口径的**几何判据**（不看分数）。

方法沿用项目既有做法（`_k_scale_diag.py` 的同源思路）：拿**交付件**（0.7963）的框
当独立参照，只比几何一致性。

比较两个候选解释：
    A) panel  —— gate 输出的 x 直接当可见光格坐标（即提示词的字面要求）
    B) whole  —— gate 输出的 x 是**整张三联图**的坐标 ⇒ x 乘 3 换回可见光格

判据：与交付件框的 IoU。哪个解释的 IoU 系统性更高，哪个就是模型的真实输出约定。
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "_m_gate20_v1.json")
REF = os.path.join(HERE, "submission_zips", "semifinal", "0.7963",
                   "result_semi_q38_nonord.json")


def as_box(v):
    """兼容 [x1,y1,x2,y2] / {"bbox":[...]} 两种写法；取归一化 0-1。"""
    if isinstance(v, dict):
        v = v.get("bbox") or v.get("bbox_2d")
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


def x3(bb):
    v = [min(bb[0] * 3.0, 1.0), bb[1], min(bb[2] * 3.0, 1.0), bb[3]]
    return [min(v[0], v[2]), min(v[1], v[3]), max(v[0], v[2]),
            max(v[1], v[3])]


def main() -> int:
    g = json.load(open(GATE, encoding="utf-8"))
    r = json.load(open(REF, encoding="utf-8"))
    print(f"gate 条目 {len(g)}｜交付件条目 {len(r)}")
    if isinstance(r, dict) and len(r) <= 8:
        print("★交付件条目太少，可能不是提交件结构：", list(r)[:8])

    rows = []
    for k, v in g.items():
        gb = as_box(v)
        rb = as_box(r.get(k)) if isinstance(r, dict) else None
        if gb is None:
            rows.append((k, None, None, None, "gate 无框"))
            continue
        if rb is None:
            rows.append((k, gb, None, None, "交付件无框/缺该 qid"))
            continue
        rows.append((k, gb, iou(gb, rb), iou(x3(gb), rb), ""))

    print()
    print(f"{'qid':<14}{'IoU(panel)':>11}{'IoU(whole)':>11}   判定")
    pa, wh = [], []
    for k, gb, ia, ib, note in rows:
        if ia is None or ib is None:
            print(f"{k:<14}{'—':>11}{'—':>11}   {note}")
            continue
        pa.append(ia)
        wh.append(ib)
        flag = "whole 更近" if ib > ia + 1e-9 else (
            "panel 更近" if ia > ib + 1e-9 else "持平")
        print(f"{k:<14}{ia:>11.4f}{ib:>11.4f}   {flag}")
    print()
    if pa:
        print(f"  可比条数 {len(pa)}")
        print(f"  panel：中位 IoU {st.median(pa):.4f}｜均值 {st.mean(pa):.4f}"
              f"｜>0.5 的 {sum(1 for x in pa if x > 0.5)} 条")
        print(f"  whole：中位 IoU {st.median(wh):.4f}｜均值 {st.mean(wh):.4f}"
              f"｜>0.5 的 {sum(1 for x in wh if x > 0.5)} 条")
        print()
        if st.median(wh) > st.median(pa) + 0.02:
            print("  ⇒ ★判 whole（x 乘 3）：置 QWEN_MERGE_SPACE=whole")
        elif st.median(pa) > st.median(wh) + 0.02:
            print("  ⇒ ★判 panel（不换算）：维持 QWEN_MERGE_SPACE=panel")
        else:
            print("  ⇒ 两者接近，判据不足 ⇒ 看 G3 拼版图人眼定夺")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
