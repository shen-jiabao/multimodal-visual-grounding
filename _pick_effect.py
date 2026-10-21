#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""估计 pick_v1 相对 annotate 的有效改动面：54 处贴蓝按 IoU 分层。

IoU>=0.5 ⇒ 两件同侧，ACC@0.5 下几乎同对同错 ⇒ 翻动无效；
IoU<0.5 ⇒ 真分歧位，每处期望净收益 = (2p-1)，p=人眼判对率。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if (ar + br - inter) > 0 else 0.0


def main() -> int:
    blue = json.loads((R / "submission_zips/semifinal/0.7963/"
                       "result_semi_q38_nonord.json").read_text("utf-8"))
    red = json.loads(zipfile.ZipFile(
        R / "submission_zips/semifinal/未提交/"
        "result_semi_27b_annotate.zip").read("submission.json").decode())
    ch = json.loads((R / "_pick_choices.json").read_text("utf-8"))
    blues = [k for k, v in ch.items() if v == "blue"]
    eff = [k for k in blues if _iou(blue[k]["bbox"], red[k]["bbox"]) < 0.5]
    zero = [k for k in eff if _iou(blue[k]["bbox"], red[k]["bbox"]) == 0]
    print(f"贴蓝 {len(blues)}｜有效分歧位(IoU<0.5) {len(eff)}"
          f"（其中 IoU=0 {len(zero)}）｜无效高重叠 {len(blues)-len(eff)}")
    n = 5690
    for p in (0.5, 0.6, 0.7, 0.8):
        gain = len(eff) * (2 * p - 1)
        print(f"  判对率 {p:.0%} ⇒ 净 +{gain:.1f} 条 ⇒ "
              f"+{gain/n:.4f}（annotate 期望之上）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
