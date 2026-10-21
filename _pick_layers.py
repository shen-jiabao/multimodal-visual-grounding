#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按差异位 IoU 分层，给出"人眼选择"的战场规模与时间预算。"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
BLUE = R / "submission_zips" / "semifinal" / "0.7963" / "result_semi_q38_nonord.json"
RED_ZIP = R / "submission_zips" / "semifinal" / "未提交" / "result_semi_27b_annotate.zip"


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
    diff = sorted(k for k in blue if blue[k]["bbox"] != red[k]["bbox"])
    buckets = {"iou0": [], "lt03": [], "03_05": [], "05_09": [], "ge09": []}
    for k in diff:
        v = _iou(blue[k]["bbox"], red[k]["bbox"])
        if v == 0:
            buckets["iou0"].append(k)
        elif v < 0.3:
            buckets["lt03"].append(k)
        elif v < 0.5:
            buckets["03_05"].append(k)
        elif v < 0.9:
            buckets["05_09"].append(k)
        else:
            buckets["ge09"].append(k)
    for name, ks in buckets.items():
        print(f"{name:7s} {len(ks):5d}")
    # 人眼选择的作用域：IoU<0.5（两件不同目标/明显不同框）
    battle = sorted(buckets["iou0"] + buckets["lt03"] + buckets["03_05"])
    (R / "_pick_battle.json").write_text(
        json.dumps(battle, ensure_ascii=False), "utf-8")
    print(f"\n战场（IoU<0.5，需人眼）：{len(battle)} 条"
          f"｜@15s/条 ≈ {len(battle)*15/3600:.1f} 小时")
    hi = sorted(buckets["05_09"] + buckets["ge09"])
    print(f"高重叠（IoU>=0.5，默认红不选）：{len(hi)} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
