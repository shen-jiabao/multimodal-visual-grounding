# -*- coding: utf-8 -*-
"""序数 SoM 精确对错判定：按问句方向推算期望编号，与实际选中编号对比。

原理：枚举提示词固定输出「从左到右」排序的候选表，编号即位置。因此
    LTR（from left to right / leftmost / from the left）: 期望 num = k
    RTL（from right to left / rightmost / from the right）: 期望 num = n_cands - k + 1
若实际 num 与期望不符，说明模型选中的候选与问句要求的实例位置不一致
（前提：模型遵守编号约定——它被明确问"是哪个编号"）。

注意「大概率对」与「必对」的区别：num == 期望 只表明编号语义自洽，
候选表第 k 个未必就是目标实例（枚举可能漏了实例或顺序有偏）。

输出分类：
    BEYOND      k > n_cands      → 必错（已由 SOM_UNDERCOVER 拦截）
    MATCH       num == 期望       → 编号自洽（大概率对）
    MISMATCH    num != 期望       → 位置不符（大概率错）
    NO_DIR      问句无方向词       → 无法判定
    NO_K        问句无序数词       → 无法判定
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
# 方向/参照物/候选有序性判据的唯一实现在 `_fuse_qwen`，此处只做转发，
# 避免两份判据漂移（本项目已因双份判据出过一次错）。
from _fuse_qwen import direction  # noqa: E402,F401

RTL_PAT = F.DIR_RTL
LTR_PAT = F.DIR_LTR


def main() -> int:
    som = json.loads((R / "qwen3vl_som_all.json").read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))

    buckets = defaultdict(list)
    dir_stat = Counter()
    for qid, rec in som.items():
        if not rec.get("bbox"):
            continue
        nc = rec.get("n_cands") or 0
        num = rec.get("num")
        q = test.get(qid, {}).get("query", "")
        k = F.ordinal_index(q)
        d = direction(q)
        dir_stat[d or "NONE"] += 1
        if k is None:
            buckets["NO_K"].append((qid, k, num, nc, d))
            continue
        if k > nc:
            buckets["BEYOND"].append((qid, k, num, nc, d))
            continue
        if d == "":
            buckets["NO_DIR"].append((qid, k, num, nc, d))
            continue
        exp = k if d == "LTR" else nc - k + 1
        key = "MATCH" if num == exp else "MISMATCH"
        buckets[key].append((qid, k, num, nc, d, exp))

    print("方向分布（已采纳 SoM 的条目）:", dict(dir_stat))
    tot = sum(len(v) for v in buckets.values())
    print(f"已采纳 SoM 且有框的条目: {tot}\n")
    for key in ("BEYOND", "MISMATCH", "MATCH", "NO_DIR", "NO_K"):
        v = buckets[key]
        pct = len(v) / max(tot, 1)
        print(f"  {key:9s} {len(v):5d}  ({pct:5.1%})")
    print()
    decidable = len(buckets["MATCH"]) + len(buckets["MISMATCH"])
    if decidable:
        print(f"可判定（方向+序号均明确）: {decidable}")
        print(f"  编号自洽 MATCH    {len(buckets['MATCH'])} "
              f"({len(buckets['MATCH']) / decidable:.1%})")
        print(f"  位置不符 MISMATCH {len(buckets['MISMATCH'])} "
              f"({len(buckets['MISMATCH']) / decidable:.1%})")
    print()
    print("MISMATCH 样例（期望 vs 实际）:")
    for r in buckets["MISMATCH"][:18]:
        qid, k, num, nc, d, exp = r
        print(f"   {qid}  {d} k={k} n_cands={nc} 期望num={exp} 实际num={num}"
              f"   {test[qid]['query'][:62]!r}")
    print()
    print("BEYOND 样例（k > n_cands，必错，已被拦）:")
    for r in buckets["BEYOND"][:10]:
        qid, k, num, nc, d = r
        print(f"   {qid}  {d} k={k} n_cands={nc} 实际num={num}"
              f"   {test[qid]['query'][:62]!r}")

    out = {k: [list(x) for x in v] for k, v in buckets.items()}
    (R / "_som_ordinal_verdict.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细 → {R / '_som_ordinal_verdict.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())