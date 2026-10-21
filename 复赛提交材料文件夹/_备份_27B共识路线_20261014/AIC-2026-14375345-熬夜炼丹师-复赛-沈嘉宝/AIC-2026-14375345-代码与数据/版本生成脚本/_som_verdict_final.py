# -*- coding: utf-8 -*-
"""最终口径：可靠子集（bbox_all 严格有序）内的 SoM 序数判定。

三重约束叠加，只保留可辩护的条目：
1. bbox_all 严格按 x 升序（num→位置映射可靠）
2. 问句方向明确（LTR/RTL）且序号 k <= n_cands
3. 序数词不是「方位介词引导的参照物」

参照物只识别方位关系（"to the right of the second X"、"next to the first X"、
"between the first and second X"），**不含** "of the"——后者多为
"the trunk of the second tree" 这类「序数仍是目标标识」的属格结构，剔除会
误伤真 MISMATCH（已在 000870_003 / 003614_003 / 009669_002 上验证）。
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
from _analyze_som_ordinal import direction  # noqa: E402

# 仅方位介词（不含 of the 属格）
REF_SPATIAL = re.compile(
    r"(to\s+the\s+(right|left)\s+of|"
    r"(right|left)[-\s](rear|front)?\s*of\s+the|"
    r"behind\s+the|in\s+front\s+of\s+the|next\s+to\s+the|beside\s+the|"
    r"near\s+the|between\s+the|above\s+the|below\s+the|under\s+the|"
    r"on\s+top\s+of\s+the|adjacent\s+to\s+the)\s*$", re.I)
WIN = 32
TOL = 0.02


def ref_target(query: str) -> bool:
    """序数词是否被方位介词引导而成为参照物（目标为别的东西）。"""
    if not query:
        return False
    for m in F.ORD_PAT.finditer(query):
        if REF_SPATIAL.search(query[max(0, m.start() - WIN):m.start()]):
            return True
    return False


def strict_sorted(rec: dict) -> bool:
    """候选是否严格按 x 升序（容差内无逆序）。"""
    c = rec.get("bbox_all") or []
    if len(c) < 2:
        return False
    cx = [(b[0] + b[2]) / 2 for b in c]
    return all(cx[i] <= cx[i + 1] + TOL for i in range(len(cx) - 1))


def main() -> int:
    som = json.loads((R / "qwen3vl_som_all.json").read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))

    stat = Counter()
    rows = []
    for qid, rec in som.items():
        if not rec.get("bbox"):
            stat["no_bbox"] += 1
            continue
        q = test.get(qid, {}).get("query", "")
        nc = rec.get("n_cands") or 0
        num = rec.get("num")
        k = F.ordinal_index(q)
        d = direction(q)
        if k is None:
            stat["no_ordinal"] += 1
            continue
        if d == "":
            stat["no_direction"] += 1
            continue
        if k > nc:
            stat["beyond_n_cands"] += 1
            continue
        if not strict_sorted(rec):
            stat["unsorted_cands"] += 1
            continue
        if ref_target(q):
            stat["ordinal_is_reference"] += 1
            continue
        exp = k if d == "LTR" else nc - k + 1
        v = "MATCH" if num == exp else "MISMATCH"
        stat[v] += 1
        rows.append({"qid": qid, "d": d, "k": k, "nc": nc, "num": num,
                     "exp": exp, "verdict": v, "query": q,
                     "core": rec.get("core")})

    tot = stat["MATCH"] + stat["MISMATCH"]
    print("=== 最终可靠口径（三重约束后可判定）===")
    print(f"可判定 {tot} 条")
    print(f"  MATCH    {stat['MATCH']:5d}  ({stat['MATCH'] / max(tot, 1):.1%})")
    print(f"  MISMATCH {stat['MISMATCH']:5d}  ({stat['MISMATCH'] / max(tot, 1):.1%})")
    print()
    print("排除项（不计入判定）:")
    for k in ("unsorted_cands", "ordinal_is_reference", "beyond_n_cands",
              "no_direction", "no_ordinal", "no_bbox"):
        print(f"  {k:22s} {stat[k]}")
    print()
    print(f"  {'方向':5s} {'MATCH':>7} {'MISMATCH':>9} {'差错率':>8}")
    for d in ("LTR", "RTL"):
        m = sum(1 for r in rows if r["d"] == d and r["verdict"] == "MATCH")
        x = sum(1 for r in rows if r["d"] == d and r["verdict"] == "MISMATCH")
        print(f"  {d:5s} {m:>7} {x:>9} {x / max(m + x, 1):>7.1%}")

    mm = [r for r in rows if r["verdict"] == "MISMATCH"]
    print()
    print(f"MISMATCH 涉及组数: "
          f"{len({(r['qid'].split('_')[0], r['core']) for r in mm})}")
    print("MISMATCH 按 n_cands 分布:",
          dict(sorted(Counter(r["nc"] for r in mm).items())))
    print("MISMATCH 按 k 分布:",
          dict(sorted(Counter(r["k"] for r in mm).items())))

    (R / "_som_verdict_final.json").write_text(
        json.dumps({"stat": dict(stat), "rows": rows,
                    "mismatch_qids": sorted(r["qid"] for r in mm)},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细 → {R / '_som_verdict_final.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())