#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记录第二批判定 + 统计共识强度分层 + 与 pick_v1 已判蓝位重叠。"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
ks = json.loads((R / "_consensus_annonly.json").read_text("utf-8"))
sp = json.loads((R / "_consensus_split.json").read_text("utf-8"))

# 第二批：全局 21..40 中，批内序号 4/12/16 为红
red2 = {24, 32, 36}
p = R / "_consensus_calib.json"
calib = json.loads(p.read_text("utf-8")) if p.exists() else {}
for g in range(21, 41):
    calib[ks[g - 1]] = "red" if g in red2 else "blue"
p.write_text(json.dumps(calib, ensure_ascii=False, indent=1), "utf-8")

# 命中率（共识预测恒为 blue）
n = len(calib)
hit = sum(1 for v in calib.values() if v == "blue")
print(f"已判 {n} 条｜共识命中 {hit}（{hit/n*100:.1f}%）｜"
      f"若共识无信息则期望 {n/2:.0f}")

# 共识强度分布
vs = sorted((r["vote_q38"] for r in sp["annotate_only"]), reverse=True)
print(f"\n333 条 vote_q38 分布：max={vs[0]} min={vs[-1]}")
for lo in (30, 25, 20, 15, 10, 5):
    print(f"  >= {lo:>2} 票：{sum(1 for v in vs if v >= lo)} 条")
print(f"前 40 条票数范围 {vs[0]} ~ {vs[39]}")

# 与 pick_v1 已判蓝位重叠
ch = json.loads((R / "_pick_choices.json").read_text("utf-8"))
blue_old = {k for k, v in ch.items() if v == "blue"}
ann_only = {r["qid"] for r in sp["annotate_only"]}
print(f"\npick_v1 已判蓝 {len(blue_old)} 处｜其中落在本 333 条内："
      f"{len(blue_old & ann_only)} 处")
print(f"333 条中尚未被用户判过的：{len(ann_only - set(ch))} 条")
