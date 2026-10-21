# -*- coding: utf-8 -*-
"""SoM 核心词事后审计：按核心词分组统计枚举成功率，定位失败集中区。

数据到齐后运行，回答三个问题：
1. 哪些核心词枚举不出来（n_cands<=1）——提示词适配度问题
2. **枚举召回率**：组内问句数 N 就等于数据集认为该图有几个实例（问"第一/
   二/三个白花盆"⇒ ≥3 个），故 `n_cands` vs `N` 是可直接量的召回率。
   比 `n_cands<=1` 的粗阈值严格更有信息量——能抓出「N=5 只枚举出 3」这类
   被标 `ok` 却会静默选错实例的条目。
3. `_fuse_qwen.untrusted_ordinals()` 的判据是否被实测证实——若被判「不可信」
   的条目在生产数据里真的枚举失败，则判据成立；若它们反而返回多候选，那正是
   想规避的风险，必须复核。

输入：qwen3vl_som_all.json + 核心词表 + test.json（取图名以分组）
输出：控制台表格 + _som_core_audit.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402

SOM_JSON = R / "qwen3vl_som_all.json"
OUT = R / "_som_core_audit.json"
TOP = 40


def recall_report(som: dict, cores: dict, test: dict) -> dict:
    """枚举召回率：按 (图, 核心词) 组比较 n_cands 与组内问句数 N。

    N 就是数据集自己声明的实例数下限（问 1/2/3 个白花盆 ⇒ 该图至少 3 个），
    因此 n_cands < N 即欠枚举。重点是被标 `ok` 却欠枚举的组——那类会静默选到
    错实例，比 `enum_fail`（安全回退）危险。

    返回统计字典，同时打印控制台明细。
    """
    grp: dict = defaultdict(list)
    for qid, c in cores.items():
        img = Path(test[qid]["visible"]).name if qid in test else "?"
        grp[(img, c.lower())].append(qid)

    rows = []
    for key, qids in grp.items():
        v = som.get(qids[0])
        if not v:
            continue
        rows.append({"img": key[0], "core": key[1], "N": len(qids),
                     "n_cands": v.get("n_cands") or 0,
                     "reason": v.get("reason", "ok"), "qids": sorted(qids)})
    short = [r for r in rows if r["n_cands"] < r["N"]]
    silent = [r for r in short if r["reason"] == "ok"]
    got = [q for r in silent for q in r["qids"]
           if som.get(q, {}).get("bbox")]

    print(f"\n== 枚举召回率 ==")
    print(f"已枚举组 {len(rows)}；欠枚举(n_cands<N) {len(short)}"
          f"（{len(short) / max(len(rows), 1):.1%}），覆盖条目 "
          f"{sum(r['N'] for r in short)}")
    dist = Counter((r["N"], r["n_cands"]) for r in rows)
    print(f"(N, n_cands) 分布: {dict(sorted(dist.items()))}")
    print(f"*** 欠枚举却标 ok 的组 {len(silent)}，覆盖条目 "
          f"{sum(r['N'] for r in silent)}，其中已产出 bbox {len(got)}")
    if got:
        print(f"    [需复核] 可能静默选错实例: {sorted(got)[:12]}")
    for r in sorted(short, key=lambda x: x["n_cands"] - x["N"])[:20]:
        print(f"  {r['img']} {r['core']!r:22s} N={r['N']} "
              f"n_cands={r['n_cands']} reason={r['reason']} {r['qids']}")
    return {"groups": len(rows), "short": len(short), "silent": len(silent),
            "silent_qids": sorted(got), "rows": rows}


def main() -> int:
    if not SOM_JSON.exists():
        print(f"[待运行] 尚未拉回 {SOM_JSON.name}")
        return 2
    som = json.loads(SOM_JSON.read_text(encoding="utf-8"))
    cores = json.loads(F.CORES_JSON.read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    untrusted = F.untrusted_ordinals()

    print(f"SoM 记录 {len(som)} / 序数问句 {len(cores)}")
    miss = set(cores) - set(som)
    if miss:
        print(f"[警告] 缺失 {len(miss)} 条，样例 {sorted(miss)[:5]}")

    reasons = Counter(v.get("reason", "ok") for v in som.values())
    hit = sum(1 for v in som.values() if v.get("bbox"))
    enum_ok = sum(1 for v in som.values() if (v.get("n_cands") or 0) > 1)
    n = max(len(som), 1)
    print(f"总体: 有框 {hit} ({hit / n:.1%})  "
          f"枚举出多候选 {enum_ok} ({enum_ok / n:.1%})")
    print(f"原因分布: {dict(reasons)}")

    # 逐核心词统计（按 (核心词) 聚合，记录 qid 以便回溯）
    by_core: dict = defaultdict(list)
    for qid, v in som.items():
        by_core[(v.get("core") or cores.get(qid) or "?").lower()].append(
            (qid, v))
    rows = []
    for core, pairs in by_core.items():
        m = len(pairs)
        ok = sum(1 for _, v in pairs if (v.get("n_cands") or 0) > 1)
        h = sum(1 for _, v in pairs if v.get("bbox"))
        rows.append({
            "core": core,
            "n": m,
            "enum_ok": ok,
            "hit": h,
            "enum_rate": round(ok / m, 3),
            "hit_rate": round(h / m, 3),
            "avg_cands": round(sum(v.get("n_cands") or 0
                                   for _, v in pairs) / m, 2),
            "reasons": dict(Counter(v.get("reason", "ok") for _, v in pairs)),
            "qids": sorted(q for q, _ in pairs),
        })
    rows.sort(key=lambda r: (r["enum_rate"], -r["n"]))

    print(f"\n核心词 {len(rows)} 个；枚举率最低的 {TOP} 个：")
    print(f"  {'核心词':34s} {'n':>4} {'枚举率':>7} {'命中率':>7}  原因")
    for r in rows[:TOP]:
        print(f"  {r['core'][:34]:34s} {r['n']:>4} {r['enum_rate']:>7.2f} "
              f"{r['hit_rate']:>7.2f}  {r['reasons']}")

    # 判据验证
    u_pairs = [(q, v) for q, v in som.items() if q in untrusted]
    u_enum_ok = [q for q, v in u_pairs if (v.get("n_cands") or 0) > 1]
    u_hit = sum(1 for _, v in u_pairs if v.get("bbox"))
    print(f"\n判据验证：不可信集合 {len(untrusted)} 条（已产出 {len(u_pairs)}）")
    print(f"  枚举出多候选 {len(u_enum_ok)}  有框 {u_hit}")
    print(f"  原因 {dict(Counter(v.get('reason', 'ok') for _, v in u_pairs))}")
    if u_enum_ok:
        print("  [需复核] 以下被判不可信却返回了多候选（风险真实存在）：")
        print(f"    {sorted(u_enum_ok)[:12]}")

    recall = recall_report(som, cores, test)

    OUT.write_text(json.dumps({
        "total": len(som), "hit": hit, "enum_ok": enum_ok,
        "reasons": dict(reasons), "by_core": rows,
        "recall": {k: v for k, v in recall.items() if k != "rows"},
        "recall_rows": recall["rows"],
        "untrusted_check": {
            "n_untrusted": len(untrusted),
            "produced": len(u_pairs),
            "enum_ok": len(u_enum_ok),
            "hit": u_hit,
            "multicand_qids": sorted(u_enum_ok),
        },
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n明细 → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
