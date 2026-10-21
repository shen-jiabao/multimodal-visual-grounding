# -*- coding: utf-8 -*-
"""检查：非序数层里有没有漏掉空间比较类 query（0 GPU，纯 CPU）。

问题来源
    已实测 VLX 在空间比较类 query 上会错（冒烟 `000085_004` 第三栋=第二栋、
    `000113_004` 漏到 obj67）。若序数分类器把这类题漏进了非序数层，
    那么 F 就是把这些题喂给 VLX 最不擅长的能力，而改动率一高伤害直接。
    故在开跑前把规模量出来 —— 它决定全量读数怎么读。

两套口径（都是为了看敏感性，不预设哪个对）
    A 严格 = 用户给出的词表（超级/比较级，最可能漏检的那批）
    B 宽松 = A + 方位轴短语（"from left to right" 等），序数层的典型句式

用法
    python -B _n32_spatial_gap.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"

PAT_A = re.compile(
    r"\b(leftmost|rightmost|topmost|bottommost|frontmost|rearmost|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"last|nearest|closest|farthest|furthest)\b", re.I)

PAT_B = re.compile(
    r"(\b(leftmost|rightmost|topmost|bottommost|frontmost|rearmost|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"last|nearest|closest|farthest|furthest|nth|middle)\b|"
    r"left\s+to\s+right|right\s+to\s+left|"
    r"from\s+the\s+left|from\s+the\s+right|"
    r"\b\d+(st|nd|rd|th)\b)", re.I)


def main() -> int:
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    core = json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8"))
    ord_ids = set(core if isinstance(core, list) else core.keys())

    print(f"queries.json 结构：{type(q).__name__}，共 {len(q)} 条")
    print(f"序数核心集 {len(ord_ids)} 条（归规则管，本检查跳过）")

    items = q.items() if isinstance(q, dict) else [(r["id"], r) for r in q]
    nonord = [(k, v) for k, v in items if k not in ord_ids]
    print(f"非序数条目总数：{len(nonord)}\n")

    for tag, pat in (("A 严格（用户词表）", PAT_A), ("B 宽松（+方位轴短语）", PAT_B)):
        hits = []
        for qid, v in nonord:
            text = v.get("query") or v.get("text") or ""
            m = pat.findall(text)
            if m:
                toks = sorted({(x[0] if isinstance(x, tuple) else x).lower()
                               for x in m})
                hits.append((qid, toks, text))
        rate = len(hits) / max(len(nonord), 1)
        print(f"=== {tag} ===")
        print(f"  命中 {len(hits)} / {len(nonord)} = {rate:.1%}")
        cnt = Counter(t for _, ts, _ in hits for t in ts)
        print(f"  词频 top12：{cnt.most_common(12)}")
        for qid, toks, text in hits[:15]:
            print(f"    {qid}  {toks}  {text[:90]}")
        print()
        if tag.startswith("A"):
            (R / "_n32_spatial_gap.json").write_text(
                json.dumps([h[0] for h in hits], ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"  已写 _n32_spatial_gap.json（{len(hits)} 条）\n")

    # 抽看：这批命中里有多少是"同图多实例"句式（最容易被误分的）
    print("=== 抽样：命中里含 'left to right' / 'right to left' 的句式 ===")
    axis = re.compile(r"(left\s+to\s+right|right\s+to\s+left)", re.I)
    n_axis = sum(1 for _, v in nonord if axis.search(v.get("query") or ""))
    print(f"  非序数层中含方位轴短语的：{n_axis} 条")
    shown = 0
    for qid, v in nonord:
        t = v.get("query") or ""
        if axis.search(t) and shown < 10:
            print(f"    {qid}  {t[:88]}")
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
