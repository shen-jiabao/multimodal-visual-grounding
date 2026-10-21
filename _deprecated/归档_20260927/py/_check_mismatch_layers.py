# -*- coding: utf-8 -*-
"""决定性检验：在 `bbox_all` 严格有序的可靠子集里重算 MISMATCH。

动机：期望编号公式依赖「候选按 x 从左到右排列」。全量 1641 条多候选记录中
250 条（15.2%）存在逆序。若 MISMATCH 主要集中在这些不可靠记录里，则 333
这个数字是判据自身的产物而非模型的错误。

做法：按逆序严重度分层，逐层重算 MATCH/MISMATCH 与 RTL/LTR 对比。
- 严格有序：无任何逆序
- 轻微逆序：最大回退 <= 0.02（两个候选几乎同 x，可能是重复检出）
- 严重逆序：最大回退 > 0.02

另打印 20 条 MISMATCH 全问句，供人工核对方向与期望编号。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
from _analyze_som_ordinal import direction  # noqa: E402

TOL = 0.02


def severity(cx: list) -> tuple:
    """返回 (最大回退幅度, 逆序处数)。"""
    drops = [cx[i] - cx[i + 1] for i in range(len(cx) - 1) if cx[i] > cx[i + 1]]
    return (max(drops) if drops else 0.0), len(drops)


def main() -> int:
    som = json.loads((R / "qwen3vl_som_all.json").read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))

    # 先给每条记录定「有序等级」
    grade = {}
    for qid, rec in som.items():
        cands = rec.get("bbox_all") or []
        if len(cands) < 2:
            grade[qid] = "lt2"
            continue
        cx = [(b[0] + b[2]) / 2 for b in cands]
        drop, n_inv = severity(cx)
        grade[qid] = ("strict" if n_inv == 0
                      else "slight" if drop <= TOL else "severe")

    print("记录有序等级分布（全部 som 记录）:")
    print("  ", dict(Counter(grade.values())))

    # 重算判定，按等级分层
    layers = defaultdict(lambda: {"MATCH": Counter(), "MISMATCH": Counter()})
    details = []
    for qid, rec in som.items():
        if not rec.get("bbox"):
            continue
        nc = rec.get("n_cands") or 0
        num = rec.get("num")
        q = test.get(qid, {}).get("query", "")
        k = F.ordinal_index(q)
        d = direction(q)
        if k is None or k > nc or d == "":
            continue
        exp = k if d == "LTR" else nc - k + 1
        key = "MATCH" if num == exp else "MISMATCH"
        g = grade.get(qid, "na")
        layers[g][key][d] += 1
        if key == "MISMATCH":
            details.append({"qid": qid, "grade": g, "d": d, "k": k,
                            "nc": nc, "exp": exp, "num": num,
                            "query": q})

    print()
    print("按 bbox_all 有序等级分层重算（只统计方向+序号均可判定的条目）:")
    print(f"  {'等级':8s} {'LTR M/X':>16s} {'RTL M/X':>16s} "
          f"{'总MISMATCH率':>13s} {'RTL率':>8s}")
    for g in ("strict", "slight", "severe"):
        c = layers[g]
        lm, lx = c["MATCH"]["LTR"], c["MISMATCH"]["LTR"]
        rm, rx = c["MATCH"]["RTL"], c["MISMATCH"]["RTL"]
        tot_m, tot_x = lm + rm, lx + rx
        rate = tot_x / max(tot_m + tot_x, 1)
        rrate = rx / max(rm + rx, 1)
        print(f"  {g:8s} {f'{lm}/{lx}':>16s} {f'{rm}/{rx}':>16s} "
              f"{rate:>12.1%} {rrate:>7.1%}")

    strict_mm = [d for d in details if d["grade"] == "strict"]
    print()
    print(f"可靠子集（strict）里的 MISMATCH: {len(strict_mm)} 条 "
          f"/ 全部 MISMATCH {len(details)} 条 "
          f"= {len(strict_mm) / max(len(details), 1):.1%}")

    print()
    print("=" * 72)
    print("20 条 MISMATCH 全问句（人工核对：方向、序号、期望编号）")
    print("=" * 72)
    for d in details[:20]:
        print(f"[{d['grade']:6s}] {d['qid']} {d['d']} k={d['k']} nc={d['nc']}"
              f" 期望num={d['exp']} 实际num={d['num']}")
        print(f"         {d['query']}")

    print()
    print("MISMATCH 的 grade 分布:", dict(Counter(d["grade"] for d in details)))
    print("MISMATCH 的方向 x grade:")
    for g in ("strict", "slight", "severe"):
        sub = [d for d in details if d["grade"] == g]
        print(f"  {g:8s}", dict(Counter(d["d"] for d in sub)))

    (R / "_som_mismatch_layers.json").write_text(
        json.dumps({"grade": dict(Counter(grade.values())),
                    "details": details}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"\n明细 → {R / '_som_mismatch_layers.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())