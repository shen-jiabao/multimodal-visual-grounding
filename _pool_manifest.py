#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""反向还原投票池 voter_NN 与原始产物文件的对应关系。"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
DO = (R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
      / "AIC-2026-14375345-代码与数据" / "data_outputs")

SELF = {"result_semi_27b_annotate", "result_semi_q38_nonord",
        "result_semi_27b_jitter", "result_semi_27b_shrink",
        "result_semi_pick_v1", "result_semi_pick_v2", "result_semi_pick_v3",
        "result_semi_pick_auto", "result_semi_pick_consensus"}

with gzip.open(DO / "consensus_pool.json.gz", "rt", encoding="utf-8") as fh:
    pool = json.load(fh)


def norm(d):
    return {k: [round(float(x), 6) for x in v] for k, v in d.items()}


cands = {}
for p in sorted(SEMI.glob("**/result_*.json")):
    if "复赛提交材料" in str(p) or p.stem in SELF or p.stem in cands:
        continue
    try:
        obj = json.loads(p.read_text("utf-8"))
    except Exception:
        continue
    d = {}
    for k, v in obj.items():
        b = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(b, (list, tuple)) and len(b) == 4:
            d[str(k)] = b
    if len(d) >= 5000:
        cands[p.stem] = norm(d)

print(f"候选产物 {len(cands)} 套\n")
matched = 0
for vk in sorted(pool):
    tgt = norm(pool[vk])
    hit = [n for n, d in cands.items() if d == tgt]
    matched += 1 if hit else 0
    print(f"  {vk}  ({len(pool[vk]):>4} 条)  ←  {hit[0] if hit else '未匹配'}")
print(f"\n匹配成功 {matched}/{len(pool)}")
