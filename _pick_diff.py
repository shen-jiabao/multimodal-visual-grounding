#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""算 q38 件（蓝，0.7963）与 annotate 件（红）的差异位清单。

只有 bbox 不同的位置才需要人眼选择；相同位置两件一致、无需看。
输出 `_pick_qids.json` = 按 qid 排序的差异位列表（选择协议的作用域）。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
BLUE = R / "submission_zips" / "semifinal" / "0.7963" / "result_semi_q38_nonord.json"
RED_ZIP = R / "submission_zips" / "semifinal" / "未提交" / "result_semi_27b_annotate.zip"
CORES = R / "_ordinal_cores_r2.json"


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
    blue = json.loads(BLUE.read_text("utf-8"))
    with zipfile.ZipFile(RED_ZIP) as zf:
        red = json.loads(zf.read("submission.json").decode("utf-8"))
    cores = set(json.loads(CORES.read_text("utf-8")))

    assert set(blue) == set(red), "键集不一致"
    diff = sorted(k for k in blue if blue[k]["bbox"] != red[k]["bbox"])
    same = len(blue) - len(diff)
    d_ord = [k for k in diff if k in cores]
    d_non = [k for k in diff if k not in cores]
    ious = sorted(_iou(blue[k]["bbox"], red[k]["bbox"]) for k in diff)
    n = len(ious)
    print(f"总 {len(blue)}｜一致 {same}｜差异 {len(diff)}"
          f"（序数 {len(d_ord)} / 非序数 {len(d_non)}）")
    print(f"差异位 IoU(蓝,红)：p10={ious[n//10]:.2f} p50={ious[n//2]:.2f} "
          f"p90={ious[9*n//10]:.2f}｜IoU=0 占 "
          f"{sum(1 for v in ious if v == 0)/n:.0%}｜IoU>=0.5 占 "
          f"{sum(1 for v in ious if v >= 0.5)/n:.0%}")
    (R / "_pick_qids.json").write_text(
        json.dumps(diff, ensure_ascii=False), "utf-8")
    print(f"已写 _pick_qids.json（{len(diff)} 条，qid 升序）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
