#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""战场模式辅助：列 _pick_battle.json 指定范围，标注已判/待判。"""
from __future__ import annotations

import json
import sys
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
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    q = json.loads((R / "复赛数据集-基于大模型的多模态视觉理解与推理"
                    / "queries" / "queries.json").read_text("utf-8"))
    b = json.loads((R / "submission_zips/semifinal/0.7963/"
                    "result_semi_q38_nonord.json").read_text("utf-8"))
    r = json.loads(zipfile.ZipFile(
        R / "submission_zips/semifinal/未提交/"
        "result_semi_27b_annotate.zip").read("submission.json").decode())
    ch = json.loads((R / "_pick_choices.json").read_text("utf-8")) \
        if (R / "_pick_choices.json").exists() else {}
    ks = json.loads((R / "_pick_battle.json").read_text("utf-8"))
    todo = 0
    for i, k in enumerate(ks[lo - 1:hi], lo):
        mark = ch.get(k, "")
        tag = {"blue": "已判1", "red": "已判2"}.get(mark, "待判")
        if not mark:
            todo += 1
        print(f"B{i} {k} IoU={_iou(b[k]['bbox'], r[k]['bbox']):.2f} [{tag}]  "
              f"{q[k]['query'][:55]}")
    print(f"-- 本范围待判 {todo} 条｜战场共 {len(ks)}｜已判累计 {len(ch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
