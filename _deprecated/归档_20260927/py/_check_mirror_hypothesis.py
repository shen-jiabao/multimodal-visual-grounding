# -*- coding: utf-8 -*-
"""检验「模型按自身方向计数」假设，并评估镜像修正的可行性。

背景：SEL_PROMPT 明确写「1 = leftmost, n = rightmost」。但若模型对 RTL 问句
直接返回 k（按它自己的方向数），则：
    模型答 num=k  ⟹ 真实目标是 candidates[nc-k]（0 基），而非 candidates[k-1]
即当前采纳的框是「镜像错位」，可用 nc-num+1 修正。

关键区分：
  LTR: 「按约定」与「按自身方向」两种解释给出同一个数（都是 k），故无歧义，
       其差错率（10.8%）反映真实错误。
  RTL: 两种解释不同（nc-k+1 vs k）。若 MISMATCH 里绝大多数满足 num == k，
       则说明这些是"约定分歧"而非随机错误，可行镜像修正。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402

D = json.loads((R / "_som_verdict_final.json").read_text(encoding="utf-8"))
rows = D["rows"]

print("=== 检验：MISMATCH 中 num 是否符合「按自身方向计数」=== ")
for d in ("LTR", "RTL"):
    sub = [r for r in rows if r["d"] == d]
    mm = [r for r in sub if r["verdict"] == "MISMATCH"]
    mt = [r for r in sub if r["verdict"] == "MATCH"]
    print(f"\n{d}: 可判定 {len(sub)}  MATCH {len(mt)}  MISMATCH {len(mm)}")
    if not mm:
        continue
    own = [r for r in mm if r["num"] == r["k"]]
    print(f"  MISMATCH 中 num == k（按自身方向数）: {len(own)}/{len(mm)}"
          f" = {len(own) / len(mm):.1%}")
    amb = [r for r in mm if r["nc"] == 2 * r["k"] - 1]
    own2 = [r for r in mm if r["num"] == r["k"] and r["nc"] != 2 * r["k"] - 1]
    print(f"    其中两种解释不重合（排除 nc==2k-1 的巧合）: {len(own2)}/{len(own)}")
    print(f"    nc==2k-1 导致两种解释同值: {len(amb)}")
    oth = [r for r in mm if r["num"] != r["k"]]
    print(f"  MISMATCH 中 num != k 的: {len(oth)}")
    for r in oth[:6]:
        print(f"      {r['qid']} k={r['k']} nc={r['nc']} num={r['num']} "
              f"期望{r['exp']}  {r['query'][:66]}")

print()
print("=== 镜像修正的评估（仅 RTL）===")
rtl_mm = [r for r in rows if r["d"] == "RTL" and r["verdict"] == "MISMATCH"]
mirror_ok = [r for r in rtl_mm if r["num"] == r["k"]]
print(f"RTL MISMATCH {len(rtl_mm)} 条")
print(f"  按镜像解释（真实目标 = nc-num+1）成立: {len(mirror_ok)} 条"
      f" = {len(mirror_ok) / max(len(rtl_mm), 1):.1%}")
print(f"  镜像后应取候选编号: nc - num + 1")
print()
print("  样例（模型答 num，镜像后应为 exp'）：")
for r in mirror_ok[:10]:
    nc, num = r["nc"], r["num"]
    print(f"    {r['qid']} k={r['k']} nc={nc} 模型答{num} "
          f"→ 镜像取候选{nc - num + 1}（按约定本应取{num}）")
    print(f"        {r['query'][:88]}")

print()
print("=== 风险：镜像修正与「约定遵守」条目的冲突 ===")
rtl_mt = [r for r in rows if r["d"] == "RTL" and r["verdict"] == "MATCH"]
print(f"RTL 中遵守约定的（MATCH）{len(rtl_mt)} 条 —— 对它们做镜像会改错")
print(f"  若全量 RTL 一律镜像：改对 {len(mirror_ok)}，改错 {len(rtl_mt)}")
print(f"  → 一律镜像不可取（改错多于改对）")

print()
print("=== 结论 ===")
tot_mm = len(rtl_mm) + sum(1 for r in rows if r["d"] == "LTR"
                           and r["verdict"] == "MISMATCH")
print(f"可判定 {len(rows)}：MATCH {sum(1 for r in rows if r['verdict'] == 'MATCH')}"
      f" / MISMATCH {tot_mm}")
print("RTL 差错率高（27.6% vs LTR 10.8%）的机制：模型对反向序数常按自身方向计数，")
print("未遵守「1=leftmost」约定。但仅约七成 RTL 条目出现该行为，其余遵守约定，")
print("故无法用统一规则修正——只能逐条判断或回退。")