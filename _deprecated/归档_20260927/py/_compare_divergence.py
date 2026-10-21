# -*- coding: utf-8 -*-
"""对比远端全量结果与本地部分结果差异分布，并复核关键修复样本。"""
import json
from pathlib import Path

A = Path(r"d:\Desktop\2026AiC\result_gd_3modal_v2.json")        # 远端全量
B = Path(r"d:\Desktop\2026AiC\result_gd_3modal_local_partial.json")  # 本地前 1300
a = json.loads(A.read_text(encoding="utf-8"))
b = json.loads(B.read_text(encoding="utf-8"))

common = sorted(set(a) & set(b))
rows = []
for k in common:
    ba, bb = a[k]["bbox"], b[k]["bbox"]
    d = max(abs(x - y) for x, y in zip(ba, bb))
    rows.append((d, k, ba, bb, a[k]["query"]))
rows.sort(reverse=True)

n = len(rows)
for thr in (1e-6, 1e-3, 1e-2, 5e-2, 1e-1):
    c = sum(1 for r in rows if r[0] > thr)
    print(f"差异 > {thr:g}: {c}/{n}")

print("\n=== 差异最大的 15 条 ===")
for d, k, ba, bb, q in rows[:15]:
    print(f"[{k}] diff={d:.4f}")
    print(f"   remote={[round(x,4) for x in ba]}")
    print(f"   local ={[round(x,4) for x in bb]}")
    print(f"   q={q[:80]}")

print("\n=== 关键修复样本（远端全量结果）===")
for k in ["000282_004", "000282_008", "002004_001", "002004_002"]:
    if k in a:
        print(f"{k}: {[round(x,4) for x in a[k]['bbox']]}  q={a[k]['query'][:70]}")
