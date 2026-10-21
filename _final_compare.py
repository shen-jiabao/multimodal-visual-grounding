#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终选对比：各候选件 vs annotate 基线的差异条数与构成。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
SUB = SEMI / "未提交"


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    d = ar + br - inter
    return inter / d if d > 0 else 0.0


def main() -> int:
    an = json.loads((SUB / "result_semi_27b_annotate.json").read_text("utf-8"))
    q38 = json.loads((SEMI / "0.7963" / "result_semi_q38_nonord.json")
                     .read_text("utf-8"))
    names = ["result_semi_27b_annotate", "result_semi_pick_v1",
             "result_semi_pick_v2", "result_semi_pick_v3",
             "result_semi_27b_jitter", "result_semi_27b_shrink"]
    print(f"{'件':<28}{'vs annotate':>12}{'同 q38':>9}  SHA256(前16)")
    for n in names:
        pj = SUB / f"{n}.json"
        if not pj.exists():
            continue
        d = json.loads(pj.read_text("utf-8"))
        diff = sum(1 for k in d if d[k]["bbox"] != an[k]["bbox"])
        same = sum(1 for k in d if _iou(d[k]["bbox"], q38[k]["bbox"]) >= 0.5)
        pz = SUB / f"{n}.zip"
        sha = hashlib.sha256(pz.read_bytes()).hexdigest().upper()[:16] if pz.exists() else "-"
        print(f"{n:<28}{diff:>12}{same:>9}  {sha}")

    # v3 改动构成
    v3 = json.loads((SUB / "result_semi_pick_v3.json").read_text("utf-8"))
    v1 = json.loads((SUB / "result_semi_pick_v1.json").read_text("utf-8"))
    chg = [k for k in v3 if v3[k]["bbox"] != v1[k]["bbox"]]
    cal = json.loads((R / "_consensus_calib.json").read_text("utf-8"))
    user = [k for k in chg if k in cal]
    print(f"\nv3 相对 pick_v1 改动 {len(chg)} 条｜其中你亲眼看图判过 {len(user)} 条｜"
          f"纯 32 票段默认 {len(chg)-len(user)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
