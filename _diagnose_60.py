#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：新池比旧池多采纳的那 1 条，究竟是被哪些成员"拦"下的。"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"


def bb(p: Path) -> dict:
    obj = json.loads(p.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        b = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(k)] = [float(x) for x in b]
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    d = ar + br - inter
    return inter / d if d > 0 else 0.0


# 旧池成员（含 swap 与重复项）
OLD_EXCLUDE = {"result_semi_27b_annotate", "result_semi_q38_nonord",
               "result_semi_27b_jitter", "result_semi_27b_shrink",
               "result_semi_pick_v1", "result_semi_pick_v2",
               "result_semi_pick_v3"}
members = {}
for p in sorted(SEMI.glob("**/result_*.json")):
    if "复赛提交材料" in str(p) or p.stem in OLD_EXCLUDE or p.stem in members:
        continue
    try:
        d = bb(p)
    except Exception:
        continue
    if len(d) >= 5000:
        members[p.stem] = d
print(f"旧池成员 {len(members)} 套")

main = bb(SEMI / "0.7963" / "result_semi_q38_nonord.json")  # 参照件
# 主件
for p in SEMI.parent.parent.rglob("result_semi_27b_annotate.json"):
    if "复赛提交材料" not in str(p):
        main = bb(p)
        break

# 候选位：真分歧位
cands = [k for k in main if k in main and iou(main[k], main[k]) >= 0]  # noqa
ref = bb(SEMI / "0.7963" / "result_semi_q38_nonord.json")
m = len(members)
blocked = []
for k in main:
    if k not in ref:
        continue
    if iou(main[k], ref[k]) >= 0.5:
        continue
    voters_main, voters_ref = [], []
    for n, d in members.items():
        b = d.get(k)
        if b is None:
            continue
        ia, iq = iou(b, main[k]), iou(b, ref[k])
        if ia > iq:
            voters_main.append(n)
        elif iq > ia:
            voters_ref.append(n)
    if voters_main:
        blocked.append((k, voters_main, len(voters_ref), m))

print(f"\n有「主件票」的候选位共 {len(blocked)} 处（这些位不会满票）")
cnt = {}
for k, vm, nr, tot in blocked:
    for n in vm:
        cnt[n] = cnt.get(n, 0) + 1
print("\n投过主件票的成员及其次数（前 15）：")
for n, c in sorted(cnt.items(), key=lambda x: -x[1])[:15]:
    print(f"  {c:>3} 次  {n}")
print(f"\n其中 swap 系列：",
      {n: c for n, c in cnt.items() if "swap" in n})
