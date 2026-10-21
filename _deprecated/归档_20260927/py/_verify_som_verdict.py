# -*- coding: utf-8 -*-
"""MISMATCH 判据的三重前置验证（纯本地，不碰运行产物）。

三个层次，任一层不成立则 333 这个数字不成立：

1. **排序前提**：`ENUM_PROMPT` 要求候选「ordered from left to right」，
   `SEL_PROMPT` 告知「1 = leftmost, n = rightmost」。故 `bbox_all` 必须
   真的按 x 从左到右排列，否则 num 无法映射到位置。用 bbox_all 直接验。
2. **方向解析**：打印 MISMATCH 全部问句供人工核对，并统计带连字符写法
   （left-to-right / right-to-left）是否被正则漏掉。
3. **num 偏向**：交叉验证「模型系统性偏向选小编号」假设——若成立，
   RTL 的 MISMATCH 率应显著高于 LTR。

输出：控制台报告 + _som_verdict_audit.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
from _analyze_som_ordinal import direction  # noqa: E402

SOM = R / "qwen3vl_som_all.json"
OUT = R / "_som_verdict_audit.json"

# 连字符写法（用于检查正则是否漏判）
HYPH_LTR = re.compile(r"left-to-right", re.I)
HYPH_RTL = re.compile(r"right-to-left", re.I)


def check_ordering(som: dict) -> dict:
    """检验 bbox_all 是否按 x 中心从左到右升序排列。"""
    n_ok = n_bad = n_skip = 0
    bad_samples = []
    for qid, rec in som.items():
        cands = rec.get("bbox_all") or []
        if len(cands) < 2:
            n_skip += 1
            continue
        cx = [(b[0] + b[2]) / 2 for b in cands]
        # 允许相等（框可能重叠），只要不存在明显的逆序
        inversions = sum(1 for i in range(len(cx) - 1) if cx[i] > cx[i + 1])
        if inversions == 0:
            n_ok += 1
        else:
            n_bad += 1
            if len(bad_samples) < 8:
                bad_samples.append({"qid": qid, "cx": [round(v, 3) for v in cx],
                                    "inv": inversions,
                                    "core": rec.get("core")})
    return {"sorted_ok": n_ok, "unsorted": n_bad, "lt2_cands": n_skip,
            "bad_samples": bad_samples}


def main() -> int:
    som = json.loads(SOM.read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    verdict = json.loads((R / "_som_ordinal_verdict.json").read_text(
        encoding="utf-8"))

    print("=" * 72)
    print("第 0 层：bbox_all 是否按 x 从左到右排列（期望编号的前提）")
    print("=" * 72)
    ordr = check_ordering(som)
    tot = ordr["sorted_ok"] + ordr["unsorted"]
    print(f"有 >=2 候选的记录 {tot}（另有 {ordr['lt2_cands']} 条候选<2）")
    print(f"  已排序（无逆序）{ordr['sorted_ok']}  ({ordr['sorted_ok'] / max(tot, 1):.1%})")
    print(f"  存在逆序      {ordr['unsorted']}  ({ordr['unsorted'] / max(tot, 1):.1%})")
    for b in ordr["bad_samples"]:
        print(f"    {b['qid']} {b['core']!r} cx={b['cx']} 逆序 {b['inv']} 处")

    print()
    print("=" * 72)
    print("第 1 层：方向解析审计（人工核对用）")
    print("=" * 72)
    hyph_l = [q for q, v in test.items() if HYPH_LTR.search(v["query"])]
    hyph_r = [q for q, v in test.items() if HYPH_RTL.search(v["query"])]
    print(f"含连字符 left-to-right 的问句 {len(hyph_l)}；"
          f"right-to-left {len(hyph_r)}")
    mis_l = [q for q in hyph_l if direction(test[q]["query"]) != "LTR"]
    mis_r = [q for q in hyph_r if direction(test[q]["query"]) != "RTL"]
    print(f"  连字符写法但被判错的: LTR {len(mis_l)} / RTL {len(mis_r)}")
    if mis_l or mis_r:
        print(f"    [正则漏判] LTR 例: {[(q, test[q]['query'][:50]) for q in mis_l[:3]]}")
        print(f"    [正则漏判] RTL 例: {[(q, test[q]['query'][:50]) for q in mis_r[:3]]}")
    print("  → 漏判只会落进 NO_DIR（不参与判定），不会产生假 MISMATCH")

    print()
    print("MISMATCH 全部分布（按方向）:")
    print("  ", dict(Counter(r[4] or "NONE" for r in verdict["MISMATCH"])))

    print()
    print("=" * 72)
    print("第 2 层：num 偏向假设（RTL 的 MISMATCH 率是否显著高于 LTR）")
    print("=" * 72)
    cross = defaultdict(Counter)
    for key in ("MATCH", "MISMATCH"):
        for r in verdict[key]:
            d = r[4] or "NONE"
            cross[d][key] += 1
    print(f"  {'方向':6s} {'MATCH':>7} {'MISMATCH':>9} {'MISMATCH率':>11}")
    for d in ("LTR", "RTL", "NONE"):
        c = cross[d]
        m, x = c["MATCH"], c["MISMATCH"]
        if m + x:
            print(f"  {d:6s} {m:>7} {x:>9} {x / (m + x):>10.1%}")

    print()
    print("num 取值分布（已采纳，看是否偏向小编号）:")
    nums = Counter(r[2] for key in ("MATCH", "MISMATCH")
                   for r in verdict[key])
    ncs = Counter(r[3] for key in ("MATCH", "MISMATCH")
                  for r in verdict[key])
    print("  实际 num:", dict(sorted(nums.items(), key=lambda x: (x[0] is None, x[0]))))
    print("  候选总数分布 nc:", dict(sorted(ncs.items())))
    # 归一化位置：num / nc，看是否堆积在低端
    pos = Counter()
    for key in ("MATCH", "MISMATCH"):
        for r in verdict[key]:
            if r[2] and r[3]:
                pos[round(r[2] / r[3], 1)] += 1
    print("  归一化位置 num/nc 分布:",
          dict(sorted(pos.items())))
    rtl_num1 = sum(1 for r in verdict["MISMATCH"] if r[4] == "RTL" and r[2] == 1)
    ltr_num1 = sum(1 for r in verdict["MISMATCH"] if r[4] == "LTR" and r[2] == 1)
    print(f"  MISMATCH 中 num==1: RTL {rtl_num1} 条 / LTR {ltr_num1} 条")
    print("  （若 RTL 的 num==1 很多，说明模型对『从右数』直接答了最左＝最右的反面）")

    OUT.write_text(json.dumps({
        "ordering": ordr,
        "hyphen": {"ltr": len(hyph_l), "rtl": len(hyph_r),
                   "missed_ltr": len(mis_l), "missed_rtl": len(mis_r)},
        "cross": {d: dict(c) for d, c in cross.items()},
        "num_hist": dict(sorted(nums.items(), key=lambda x: (x[0] is None, x[0]))),
        "nc_hist": dict(sorted(ncs.items())),
        "pos_hist": dict(sorted(pos.items())),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n报告 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())