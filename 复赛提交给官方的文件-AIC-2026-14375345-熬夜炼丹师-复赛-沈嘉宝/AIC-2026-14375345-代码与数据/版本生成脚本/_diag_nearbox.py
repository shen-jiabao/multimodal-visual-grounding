# -*- coding: utf-8 -*-
"""B′ 口径校验 + 「近同框」扩展候选诊断（只读，不出件）。

★上一版脚本的错误（已修）
    用 0.7766 件（B′ **已改过**）去判「基线是否等于 8B 直接框」——
    那 87 条已被换成 32B 框，自然与 8B 框对不上，于是误得「0/87」。
    B′ 脚本用的基线是 **0.7742 件**（`result_semi_n32_ordbase`），本版照此。
"""
import json
import sys
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))
import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

base = json.loads(SP.find("result_semi_n32_orddirect", ".json").read_text(encoding="utf-8"))
prev = json.loads(SP.find("result_semi_n32_ordbase", ".json").read_text(encoding="utf-8"))
cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
d32 = json.loads((R / "_qwen32b_direct_ord.json").read_text(encoding="utf-8"))
d8 = json.loads((R / "_semi_qwen3vl_all.json").read_text(encoding="utf-8"))

# ---- 1) B′ 实际改动集（两件 diff，不依赖任何推断口径）----
b_chg = [k for k in cores if base[k]["bbox"] != prev[k]["bbox"]]
print(f"[1] B′ 实际改动: {len(b_chg)} 条")

# ---- 2) 以 **0.7742 件** 为参照复原 B′ 作用域 ----
scope_b = [k for k in sorted(cores)
           if (d8.get(k) or {}).get("bbox")
           and F.iou(prev[k]["bbox"], F.norm(d8[k]["bbox"])) >= 0.999]
print(f"[2] IoU(0.7742件, 8B直接框) ≥ 0.999 → 作用域 {len(scope_b)} 条"
      f"（全部 {len(set(b_chg) & set(scope_b))} 条改动都在其中）")

# ---- 3) B′ 那 87 条改动的「判据②」剖面（此前**从未**算过）----
p = []
for k in b_chg:
    nb = F.norm(d32[k]["bbox"])
    p.append((k, F.iou(prev[k]["bbox"], nb)))
a = sorted(x[1] for x in p)
qa = lambda L, q: L[min(int(len(L) * q), len(L) - 1)]
print(f"[3] ★B′ 改动条目与(改写前)基线的 IoU: p10={qa(a,.1):.3f} p50={qa(a,.5):.3f} "
      f"p90={qa(a,.9):.3f}  ｜ <0.3 占 {sum(1 for x in a if x<0.3)/len(a):.1%}")

# ---- 4) 近同框桶（以 0.7742 件判定，排除已在 B′ 内的）----
rows = []
for k in sorted(cores):
    b8 = (d8.get(k) or {}).get("bbox")
    nb = (d32.get(k) or {}).get("bbox")
    if not b8 or not nb or k in set(scope_b):
        continue
    nb8, n32 = F.norm(b8), F.norm(nb)
    io8 = F.iou(prev[k]["bbox"], nb8)
    if not (0.9 <= io8 < 0.999):
        continue
    io32 = F.iou(prev[k]["bbox"], n32)
    rows.append((k, io8, max(abs(prev[k]["bbox"][i] - nb8[i]) for i in range(4)),
                 io32, F.valid_cand(n32)))
print(f"\n[4] 近同框桶（0.9 ≤ IoU(0.7742件, 8B) < 0.999）: {len(rows)} 条")
md = sorted(r[2] for r in rows)
print(f"    |Δ|max p50={qa(md,.5):.4f} p90={qa(md,.9):.4f} max={md[-1]:.4f}"
      f"  ｜ ≤0.003 的 {sum(1 for r in rows if r[2]<=0.003)} 条（舍入级）")

chg = [r for r in rows if r[4] and r[3] < F.IOU_KEEP]
print(f"[5] 其中 32B 会改动: {len(chg)} 条")
a2 = sorted(r[3] for r in chg)
print(f"    判据② IoU: p10={qa(a2,.1):.3f} p50={qa(a2,.5):.3f} p90={qa(a2,.9):.3f}"
      f"  ｜ <0.3 占 {sum(1 for x in a2 if x<0.3)/len(a2):.1%}")
print(f"    对照 B′: p50={qa(a,.5):.3f}, <0.3 占 {sum(1 for x in a if x<0.3)/len(a):.1%}")

# ---- 5) 与 B′ 完全同判据（≥0.999）但**未**被 B′ 覆盖的余量 ----
rest = [k for k in sorted(cores)
        if (d8.get(k) or {}).get("bbox")
        and F.iou(prev[k]["bbox"], F.norm(d8[k]["bbox"])) >= 0.999
        and k not in set(b_chg)]
print(f"\n[6] ≥0.999 且未被 B′ 改动的条目: {len(rest)} 条")
c2 = [k for k in rest if (d32.get(k) or {}).get("bbox")
      and F.valid_cand(F.norm(d32[k]["bbox"]))
      and F.iou(F.norm(d32[k]["bbox"]), prev[k]["bbox"]) < F.IOU_KEEP]
print(f"    其中 32B 会改动: {len(c2)} 条   ← B′ 的**真实余量**")
