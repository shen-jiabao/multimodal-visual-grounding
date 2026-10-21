# -*- coding: utf-8 -*-
"""对比两种口径：无护栏 vs 护栏(替换后面积>0.5 则放弃替换，保留基线)。
输出 fusion_conservative_guard.json（仅用于对比，不覆盖无护栏版本）。
"""
import json
from pathlib import Path

OUT = Path("/root/output")
L2 = json.load(open(OUT / "predictions_v2_highres_slow.json"))
M = json.load(open(OUT / "result_gd_3modal_v2.json"))
G = json.load(open(OUT / "predictions_grounding.json"))


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2-x1), max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def norm(b):
    x1, y1, x2, y2 = b
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    return [x1, y1, max(x2, x1+1e-3), max(y2, y1+1e-3)]


repl_ok, repl_rej = [], []
guard = {}
for k, rec in M.items():
    entry = dict(rec)
    if k in L2 and k in G:
        l2, m, g = L2[k]["bbox"], rec["bbox"], G[k]["bbox"]
        if iou(l2, g) >= 0.5 and iou(l2, m) < 0.5 and iou(g, m) < 0.5:
            fused = norm([0.6*l2[i] + 0.4*g[i] for i in range(4)])
            if area(fused) > 0.5:                    # 护栏：疑似整图合并框 → 不替换
                repl_rej.append(k)
            else:
                entry["bbox"] = [round(float(v), 6) for v in fused]
                repl_ok.append(k)
    guard[k] = entry

big_m = sum(1 for r in M.values() if area(r["bbox"]) > 0.35)
big_g = sum(1 for r in guard.values() if area(r["bbox"]) > 0.35)
p = OUT / "fusion_conservative_guard.json"
p.write_text(json.dumps(guard, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"无护栏: 替换 885, 大框 385 → 391 (修37 新增43)")
print(f"有护栏: 替换 {len(repl_ok)}, 放弃 {len(repl_rej)}, 大框 {big_m} → {big_g}")
print(f"被护栏拦下的样本数: {len(repl_rej)}，其中基线也是大框的: "
      f"{sum(1 for k in repl_rej if area(M[k]['bbox']) > 0.35)}")
