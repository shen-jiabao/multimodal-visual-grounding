# -*- coding: utf-8 -*-
"""统计云端新结果(result_gd_3modal_v2.json) 与 本地旧结果(result_gd_3modal.json) 的差异。"""
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
new = json.loads((R / "result_gd_3modal_v2.json").read_text(encoding="utf-8"))
old = json.loads((R / "result_gd_3modal.json").read_text(encoding="utf-8"))

print(f"云端新结果条目: {len(new)}")
print(f"本地旧结果条目: {len(old)}")
print(f"key 缺失(旧无新有): {len(set(new)-set(old))}, key 多余(旧有新无): {len(set(old)-set(new))}")

common = sorted(set(new) & set(old))
rows = []
for k in common:
    a, b = new[k]["bbox"], old[k]["bbox"]
    d = max(abs(x - y) for x, y in zip(a, b))
    rows.append((d, k))
rows.sort(reverse=True)

n = len(rows)
for thr in (1e-6, 1e-3, 1e-2, 5e-2, 1e-1):
    c = sum(1 for d, _ in rows if d > thr)
    print(f"差异 > {thr:g}: {c}/{n}  ({100*c/n:.1f}%)")

print("\n=== 变化最大的 10 条 ===")
for d, k in rows[:10]:
    print(f"[{k}] diff={d:.4f}")
    print(f"   new={[round(float(v),4) for v in new[k]['bbox']]}")
    print(f"   old={[round(float(v),4) for v in old[k]['bbox']]}")
    print(f"   q={new[k]['query'][:80]}")

# 旧结果中带 error 的数量（对比参考）
err_old = sum(1 for v in old.values() if "error" in v)
err_new = sum(1 for v in new.values() if "error" in v)
print(f"\nerror 字段: 旧={err_old}, 新={err_new}")
