# -*- coding: utf-8 -*-
"""量化「序数词作参照物」导致的 MISMATCH 误报，并给出精炼后的判定。

问题（人工核对 0023_005 时发现）：
    'The stone pier ... immediately to the right of the fourth stone pier'
    目标是「第 4 个石墩右边那个」= 第 5 个；模型答 5 正确，
    但 `ordinal_index` 取首个序数词 "fourth" -> k=4，判成 MISMATCH（误报）。

做法：识别「序数词被介词/方位词引导、实为参照物」的问句，将其从可判定
集合中剔除（改为 REF），重算 MATCH/MISMATCH 与 RTL/LTR 对比。

参照结构标记（序数词前 30 字符内出现）：
    to the right of / to the left of / right of the / left of the
    behind the / in front of the / next to the / beside the / near the
    between the / above the / below the / under the / on top of the
    of the (generic)
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

REF_MARK = re.compile(
    r"(to\s+the\s+(right|left)\s+of|"
    r"(right|left)\s+of\s+the|"
    r"behind\s+the|"
    r"in\s+front\s+of\s+the|"
    r"next\s+to\s+the|"
    r"beside\s+the|"
    r"near\s+the|"
    r"between\s+the|"
    r"above\s+the|"
    r"below\s+the|"
    r"under\s+the|"
    r"on\s+top\s+of\s+the|"
    r"\bof\s+the)\s*$",
    re.I)
WIN = 30


def is_reference(query: str) -> bool:
    """序数词是否被参照短语引导（则该序数不是目标本身）。"""
    if not query:
        return False
    for m in F.ORD_PAT.finditer(query):
        pre = query[max(0, m.start() - WIN):m.start()]
        if REF_MARK.search(pre):
            return True
    return False


def main() -> int:
    som = json.loads((R / "qwen3vl_som_all.json").read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    layers = json.loads((R / "_som_mismatch_layers.json").read_text(
        encoding="utf-8"))

    # 复现序数判定 + 参照检测
    rows = []
    for qid, rec in som.items():
        if not rec.get("bbox"):
            continue
        nc = rec.get("n_cands") or 0
        num = rec.get("num")
        q = test.get(qid, {}).get("query", "")
        k = F.ordinal_index(q)
        d = direction(q)
        ref = is_reference(q)
        if k is None or d == "" or k > nc:
            continue
        exp = k if d == "LTR" else nc - k + 1
        rows.append({"qid": qid, "k": k, "nc": nc, "d": d, "num": num,
                     "exp": exp, "ref": ref,
                     "verdict": "MATCH" if num == exp else "MISMATCH",
                     "query": q})

    print(f"可判定条目 {len(rows)}")
    n_ref = sum(1 for r in rows if r["ref"])
    print(f"  其中含「序数作参照物」结构: {n_ref} ({n_ref / max(len(rows), 1):.1%})")
    ref_mm = [r for r in rows if r["ref"] and r["verdict"] == "MISMATCH"]
    print(f"  这些条目里原本被判 MISMATCH 的: {len(ref_mm)}"
          f"（= 误报候选）")

    print()
    print("=== 精炼前后对比 ===")
    for label, sub in (("精炼前（全部可判定）", rows),
                       ("精炼后（剔除参照结构）",
                        [r for r in rows if not r["ref"]])):
        m = sum(1 for r in sub if r["verdict"] == "MATCH")
        x = sum(1 for r in sub if r["verdict"] == "MISMATCH")
        print(f"  {label}: 可判定 {len(sub)}  MATCH {m}  "
              f"MISMATCH {x} ({x / max(m + x, 1):.1%})")
        for d in ("LTR", "RTL"):
            dm = sum(1 for r in sub if r["d"] == d and r["verdict"] == "MATCH")
            dx = sum(1 for r in sub if r["d"] == d and r["verdict"] == "MISMATCH")
            if dm + dx:
                print(f"      {d}: {dm}/{dx}  差错率 {dx / (dm + dx):.1%}")

    print()
    print("参照结构判为 MISMATCH 的样例（疑似误报）:")
    for r in ref_mm[:12]:
        print(f"  {r['qid']} {r['d']} k={r['k']} nc={r['nc']} "
              f"期望{r['exp']} 实际{r['num']}")
        print(f"      {r['query'][:118]}")

    print()
    miss_ref = sum(1 for r in rows if not r["ref"] and r["verdict"] == "MISMATCH")
    print(f"结论：MISMATCH 从 {sum(1 for r in rows if r['verdict'] == 'MISMATCH')} "
          f"降至 {miss_ref}（剔除 {len(ref_mm)} 条参照结构误报）")

    (R / "_som_verdict_refined.json").write_text(
        json.dumps({"rows": rows,
                    "ref_qids": [r["qid"] for r in rows if r["ref"]],
                    "mismatch_after_refine":
                        [r["qid"] for r in rows
                         if not r["ref"] and r["verdict"] == "MISMATCH"]},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"明细 → {R / '_som_verdict_refined.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())