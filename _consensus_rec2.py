#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记录争议位第一批判定 + 统计争议位 q38 票占比分布 + 生成高占比段清单。"""
from __future__ import annotations

import collections
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
dis_list = json.loads((R / "_consensus_disputed.json").read_text("utf-8"))
red = {2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20}

p = R / "_consensus_calib.json"
cal = json.loads(p.read_text("utf-8")) if p.exists() else {}
for i in range(1, 21):
    if i - 1 < len(dis_list):
        cal[dis_list[i - 1]] = "red" if i in red else "blue"
p.write_text(json.dumps(cal, ensure_ascii=False, indent=1), "utf-8")
print(f"本批记录 20 条｜累计 {len(cal)} 条")

d = json.loads((R / "_consensus_split.json").read_text("utf-8"))
dis = d["disputed"]
b = collections.Counter()
for r in dis:
    tot = r["vote_q38"] + r["vote_annotate"]
    b[round(r["vote_q38"] / tot, 1)] += 1
print("\n争议位 q38 票占比分布（占比:条数）：")
for k in sorted(b, reverse=True):
    print(f"  {k:.1f}: {b[k]}")

hi = sorted(dis, key=lambda r: -r["vote_q38"] / (r["vote_q38"] + r["vote_annotate"]))
(R / "_consensus_disputed_hi.json").write_text(
    json.dumps([r["qid"] for r in hi], ensure_ascii=False), "utf-8")
print(f"\n高占比段前 8：")
for r in hi[:8]:
    print(f"  {r['qid']}  q38票 {r['vote_q38']} : an票 {r['vote_annotate']}")
