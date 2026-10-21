#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核准投票池 32 个成员与历史提交版本的对应（分数文件夹即实测成绩）。"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
DO = (R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
      / "AIC-2026-14375345-代码与数据" / "data_outputs")

# 1) 收集「分数文件夹 → 产物名」
scored = {}
other = {}
for d in sorted(SEMI.iterdir()):
    if not d.is_dir():
        continue
    if re.fullmatch(r"0\.\d{4}", d.name):
        for p in d.glob("result*.json"):
            scored.setdefault(p.stem, []).append(float(d.name))
    else:
        for p in d.glob("result*.json"):
            other.setdefault(p.stem, []).append(d.name)

# 2) 池成员 → 原始产物名
with gzip.open(DO / "consensus_pool.json.gz", "rt", encoding="utf-8") as fh:
    pool = json.load(fh)


def norm(d):
    return {k: [round(float(x), 6) for x in v] for k, v in d.items()}


def bb(p: Path) -> dict:
    obj = json.loads(p.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        b = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(k)] = b
    return norm(out)


cands = {}
for p in sorted(SEMI.glob("**/result*.json")):
    if "复赛提交材料" in str(p) or p.stem in cands:
        continue
    try:
        d = bb(p)
    except Exception:
        continue
    if len(d) >= 5000:
        cands[p.stem] = d

print(f"{'voter':<9}{'原始产物':<48}{'线上分数':<12}{'状态'}")
print("-" * 84)
n_scored = 0
for vk in sorted(pool):
    tgt = norm(pool[vk])
    hit = [n for n, d in cands.items() if d == tgt]
    name = hit[0] if hit else "未匹配"
    sc = scored.get(name)
    st = "历史提交·已测分" if sc else ("目录:" + (other.get(name, ["?"])[0]))
    if sc:
        n_scored += 1
    print(f"{vk:<9}{name:<48}{str(sc[0]) if sc else '—':<12}{st}")
print("-" * 84)
print(f"其中带线上分数记录的：{n_scored} / {len(pool)}")
