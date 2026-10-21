#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核查投票池中 swap 系列产物的性质：与哪个基线差异多少条。"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"


def bb(p):
    obj = json.loads(p.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        b = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(k)] = [round(float(x), 4) for x in b]
    return out


suspects = {}
for name in ("result_v2_swap33", "result_v3_swap58"):
    for p in SEMI.glob(f"**/{name}.json"):
        if "复赛提交材料" not in str(p):
            suspects[name] = bb(p)
            break
print("嫌疑产物：", {k: len(v) for k, v in suspects.items()})

others = {}
for p in sorted(SEMI.glob("**/result_*.json")):
    if "复赛提交材料" in str(p) or p.stem in suspects or p.stem in others:
        continue
    try:
        others[p.stem] = bb(p)
    except Exception:
        continue

for name, d in suspects.items():
    print(f"\n=== {name} 与各产物的差异条数（升序，前 8）===")
    diffs = []
    for n, o in others.items():
        common = [k for k in d if k in o]
        if len(common) < 5000:
            continue
        diffs.append((sum(1 for k in common if d[k] != o[k]), n))
    diffs.sort()
    for cnt, n in diffs[:8]:
        print(f"  {cnt:>5} 条  vs  {n}")
