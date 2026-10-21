#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记录 90% 段判定 + 查 annotate_only 精确票数分布 + 按纯 32 票段重算命中。"""
from __future__ import annotations

import collections
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
hi = json.loads((R / "_consensus_disputed_hi.json").read_text("utf-8"))
red = {1, 2, 5, 6, 7, 8, 10, 11, 12, 16, 17, 18, 19, 20}

p = R / "_consensus_calib.json"
cal = json.loads(p.read_text("utf-8"))
for i in range(1, 21):
    cal[hi[i - 1]] = "red" if i in red else "blue"
p.write_text(json.dumps(cal, ensure_ascii=False, indent=1), "utf-8")
print(f"本批记录 20 条｜累计 {len(cal)} 条")

sp = json.loads((R / "_consensus_split.json").read_text("utf-8"))
ao = sp["annotate_only"]
c = collections.Counter(r["vote_q38"] for r in ao)
print("\nannotate_only(333) 票数分布（票:条数）：")
for k in sorted(c, reverse=True):
    print(f"  {k:>2}: {c[k]}")

# 纯 32 票段的用户判定命中（排除争议位判定，只取独行位 40 条）
pure = [r["qid"] for r in ao if r["vote_q38"] == 32]
done = [k for k in pure if k in cal]
blue = sum(1 for k in done if cal[k] == "blue")
print(f"\n纯 32 票段 {len(pure)} 条｜其中人眼已判 {len(done)} 条｜"
      f"选蓝 {blue}（{blue/len(done)*100:.1f}%）")
