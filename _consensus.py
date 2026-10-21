#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跨产物共识分析：历史产物在 annotate vs q38 的分歧位上站哪边？

输出
  · 每套产物与 annotate 的整体一致率（IoU>=0.5 视为同框）
  · 在 596 条 IoU=0 极端分歧位上，各套产物更接近谁
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal" / "未提交"


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


def norm(obj):
    """把任意 schema 规范成 {qid: [x1,y1,x2,y2]}"""
    out = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        bb = None
        if isinstance(v, dict):
            for key in ("bbox", "box", "pred", "prediction"):
                if key in v and isinstance(v[key], (list, tuple)) and len(v[key]) == 4:
                    bb = list(map(float, v[key]))
                    break
        elif isinstance(v, (list, tuple)) and len(v) == 4 and all(
                isinstance(x, (int, float)) for x in v):
            bb = list(map(float, v))
        if bb is not None:
            out[str(k)] = bb
    return out


def main() -> int:
    an = norm(json.loads((SUB / "result_semi_27b_annotate.json").read_text("utf-8")))
    qp = SUB.parent / "0.7963" / "result_semi_q38_nonord.json"
    q38 = norm(json.loads(qp.read_text("utf-8")))
    print(f"annotate {len(an)} 条｜q38 {len(q38)} 条")

    # 分歧位分层
    common = [k for k in an if k in q38]
    zero = [k for k in common if _iou(an[k], q38[k]) == 0.0]
    lo = [k for k in common if 0.0 < _iou(an[k], q38[k]) < 0.5]
    print(f"共同位 {len(common)}｜IoU=0 {len(zero)}｜0<IoU<0.5 {len(lo)}")

    files = sorted(glob.glob(str(SUB.parent / "**" / "result_semi_*.json"),
                             recursive=True))
    files = [p for p in files if "复赛提交材料" not in p]
    rows = []
    for p in files:
        name = os.path.basename(p)
        if name.startswith("result_semi_27b_") and "annotate" not in name:
            pass  # 仍纳入参考：jitter/shrink 是衍生件
        try:
            d = norm(json.loads(Path(p).read_text("utf-8")))
        except Exception as e:
            rows.append((name, -1, -1, -1, str(e)[:30]))
            continue
        ks = [k for k in common if k in d]
        if not ks:
            rows.append((name, -1, -1, -1, "no-overlap"))
            continue
        ag = sum(1 for k in ks if _iou(d[k], an[k]) >= 0.5) / len(ks)
        qg = sum(1 for k in ks if _iou(d[k], q38[k]) >= 0.5) / len(ks)
        # 在 IoU=0 位上：更接近谁
        zk = [k for k in zero if k in d]
        z_an = z_q = 0
        for k in zk:
            ia, iq = _iou(d[k], an[k]), _iou(d[k], q38[k])
            if ia >= 0.5 and iq < 0.5:
                z_an += 1
            elif iq >= 0.5 and ia < 0.5:
                z_q += 1
        rows.append((name, ag, qg, (z_an, z_q, len(zk)), ""))

    rows.sort(key=lambda r: -r[1])
    print(f"\n{'产物':<52}{'同annotate':>10}{'同q38':>8}  IoU=0位(站an/站q38/可比)")
    for name, ag, qg, z, err in rows:
        if ag < 0:
            print(f"{name:<52}{'--':>10}{'--':>8}  {err}")
        else:
            print(f"{name:<52}{ag*100:>9.1f}%{qg*100:>7.1f}%  {z}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
