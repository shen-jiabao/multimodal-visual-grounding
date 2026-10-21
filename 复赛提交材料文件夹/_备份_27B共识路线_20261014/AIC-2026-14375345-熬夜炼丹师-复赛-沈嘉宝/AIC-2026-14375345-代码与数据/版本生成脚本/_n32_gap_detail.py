# -*- coding: utf-8 -*-
"""细化：非序数层里的空间比较类 query 到底是「序数词漏检」还是「空间最高级漏检」。

动机
    粗查得非序数层命中 646 条，但词频里 first/second/third/last 一个都没有。
    若真是如此，则结论完全不同：
      - 序数词**没有漏**（序数分类器工作正常）
      - 漏的是**空间最高级**（closest / leftmost / farthest …）—— 一类
        独立的、此前未被规则覆盖的能力
    必须分开量，否则会把"最高级"的规模记到"序数词漏检"头上。

用法
    python -B _n32_gap_detail.py
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

NUM = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|"
                 r"ninth|tenth|last|nth)\b|\b\d+(st|nd|rd|th)\b", re.I)
SUP = re.compile(r"\b(leftmost|rightmost|topmost|bottommost|frontmost|"
                 r"rearmost|nearest|closest|farthest|furthest)\b", re.I)
AXIS = re.compile(r"(left\s+to\s+right|right\s+to\s+left)", re.I)


def main() -> int:
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    core = json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8"))
    ord_ids = set(core if isinstance(core, list) else core.keys())

    nonord = [(k, v) for k, v in q.items() if k not in ord_ids]
    ordset = [(k, v) for k, v in q.items() if k in ord_ids]

    def scan(items, pat):
        out = []
        for k, v in items:
            t = v.get("query") or ""
            m = pat.findall(t)
            if m:
                toks = sorted({(x[0] if isinstance(x, tuple) else x).lower()
                               for x in m})
                out.append((k, toks, t))
        return out

    print("=" * 76)
    print("A. 序数核心集 1103 条的构成（确认分类器的实际口径）")
    print("=" * 76)
    o_num, o_sup = scan(ordset, NUM), scan(ordset, SUP)
    print(f"  含序数词 (first/second/…/last/Nth) : {len(o_num)}")
    print(f"  含空间最高级 (closest/leftmost/…)  : {len(o_sup)}")
    print(f"  两者都含                          : "
          f"{len({k for k, _, _ in o_num} & {k for k, _, _ in o_sup})}")
    print(f"  含方位轴短语 left to right 等      : {len(scan(ordset, AXIS))}")

    print()
    print("=" * 76)
    print("B. 非序数 4587 条里的漏检（切开量）")
    print("=" * 76)
    n_num = scan(nonord, NUM)
    n_sup = scan(nonord, SUP)
    n_axis = scan(nonord, AXIS)
    s_num = {k for k, _, _ in n_num}
    s_sup = {k for k, _, _ in n_sup}
    print(f"  含序数词     : {len(s_num)}  ({len(s_num)/len(nonord):.1%})")
    print(f"  含空间最高级 : {len(s_sup)}  ({len(s_sup)/len(nonord):.1%})")
    print(f"  含方位轴短语 : {len(n_axis)}")
    print(f"  并集         : {len(s_num | s_sup | {k for k,_,_ in n_axis})}")
    print()
    print("  ★ 若『含序数词』≈0 ⇒ 序数词没漏，漏的是『空间最高级』这一独立能力")
    if n_num:
        print(f"    序数词漏检样例：")
        for k, t, txt in n_num[:10]:
            print(f"      {k}  {t}  {txt[:82]}")

    print()
    print("=" * 76)
    print("C. 空间最高级的词面分布（决定是否可被现有规则吸收）")
    print("=" * 76)
    c = Counter(t for _, ts, _ in n_sup for t in ts)
    for w, n in c.most_common():
        print(f"    {w:<12} {n}")
    print()
    print("  —— closest/nearest/farthest/furthest 属『距离』语义，"
          "而深度图正是本赛题三模态之一")
    print("  —— leftmost/rightmost 属『水平轴』语义，现有序数层规则已覆盖该轴")

    print()
    print("=" * 76)
    print("D. 供全量读数分层用的清单")
    print("=" * 76)
    sup_ids = sorted(s_sup)
    (R / "_vlx_sup_split.json").write_text(
        json.dumps(sup_ids, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  已写 _vlx_sup_split.json：{len(sup_ids)} 条（含空间最高级的非序数）")
    print(f"  其余 {len(nonord) - len(sup_ids)} 条为『不含最高级』层")
    print()
    print("  分层读数口径（不新增预先设定，只加一个读表维度）：")
    print("    总分 Δ = Δ(最高级层) + Δ(其余层)")
    print("    两层的改动面与净翻转须分别报，否则一层的大亏可被另一层的")
    print("    小赚盖住（例：总 +12 = 最高级层 −40 + 其余层 +52）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
