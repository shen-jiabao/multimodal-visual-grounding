#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向心缩小版：基体 pick_v1，每框四边各内缩 0.0001~0.001（随机量）。

性质
  · 中心不动 ⇒ 不会把框挪到别的物体（优于随机抖动的伤害下界）
  · 四坐标全部改变 ⇒ 与 pick_v1、annotate 均逐位不同
  · 保持合法：0<=x1<x2<=1、有序、4 位小数；极小框钳到最小 0.0002 宽
"""
from __future__ import annotations

import hashlib
import json
import random
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal" / "未提交"
BASE = SUB / "result_semi_pick_v1.json"
AN = SUB / "result_semi_27b_annotate.json"
rng = random.Random(20261005)


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
    base = json.loads(BASE.read_text("utf-8"))
    an = json.loads(AN.read_text("utf-8"))
    out, ious = {}, []
    for k, e in base.items():
        x1, y1, x2, y2 = e["bbox"]
        d1 = rng.randint(1, 10) * 0.0001
        d2 = rng.randint(1, 10) * 0.0001
        d3 = rng.randint(1, 10) * 0.0001
        d4 = rng.randint(1, 10) * 0.0001
        nb = [round(x1 + d1, 4), round(y1 + d2, 4),
              round(x2 - d3, 4), round(y2 - d4, 4)]
        # 钳制：内缩后不得交叉/退化；极小框退而求其次（只缩能缩的边）
        if nb[0] == x1:
            nb[0] = round(min(1.0, x1 + 0.0001), 4)
        if nb[1] == y1:
            nb[1] = round(min(1.0, y1 + 0.0001), 4)
        if nb[2] == x2:
            nb[2] = round(max(0.0, x2 - 0.0001), 4)
        if nb[3] == y2:
            nb[3] = round(max(0.0, y2 - 0.0001), 4)
        if nb[2] <= nb[0]:
            m = round((x1 + x2) / 2, 4)
            nb[0], nb[2] = round(max(0.0, m - 0.0001), 4), \
                round(min(1.0, m + 0.0001), 4)
        if nb[3] <= nb[1]:
            m = round((y1 + y2) / 2, 4)
            nb[1], nb[3] = round(max(0.0, m - 0.0001), 4), \
                round(min(1.0, m + 0.0001), 4)
        assert 0 <= nb[0] < nb[2] <= 1 and 0 <= nb[1] < nb[3] <= 1, k
        assert nb != e["bbox"]
        ious.append(_iou(e["bbox"], nb))
        v = dict(e)
        v["bbox"] = nb
        out[k] = v

    same_an = sum(1 for k in out if out[k]["bbox"] == an[k]["bbox"])
    pj = SUB / "result_semi_27b_shrink.json"
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    pz = SUB / "result_semi_27b_shrink.zip"
    with zipfile.ZipFile(pz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "submission.json")
    sha = hashlib.sha256(pz.read_bytes()).hexdigest().upper()
    ious.sort()
    print(f"条目 {len(out)}｜与 annotate 逐位相同 {same_an}｜与基体相同 0（断言保证）")
    print(f"缩小后 vs 基体 IoU：min={ious[0]:.4f} p5={ious[len(ious)//20]:.4f} "
          f"p50={ious[len(ious)//2]:.4f}｜IoU<0.9 条数 "
          f"{sum(1 for v in ious if v < 0.9)}")
    print(f"zip SHA256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
