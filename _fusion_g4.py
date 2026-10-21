# -*- coding: utf-8 -*-
"""G4 定稿：替换框需同时满足 ≤0.35 且 ≤1.25×基线面积，否则保留基线。"""
import json
from collections import Counter

OUT = "/root/output"
L2 = json.load(open(f"{OUT}/predictions_v2_highres_slow.json"))
M = json.load(open(f"{OUT}/result_gd_3modal_v2.json"))
G = json.load(open(f"{OUT}/predictions_grounding.json"))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 1e-9 else 0.


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def norm(b):
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return [max(0., min(1., x1)), max(0., min(1., y1)),
            max(0., min(1., x2)), max(0., min(1., y2))]


keys = sorted(set(L2) & set(M) & set(G))
out, rules, repl_log = {}, Counter(), []
for k in keys:
    l2, m, g = L2[k]["bbox"], M[k]["bbox"], G[k]["bbox"]
    i_lm, i_lg, i_mg = iou(l2, m), iou(l2, g), iou(m, g)
    if i_lg >= 0.5 and i_lm < 0.5 and i_mg < 0.5:
        fused = norm([0.6*l2[i] + 0.4*g[i] for i in range(4)])
        a_fus, a_base = area(fused), area(m)
        if a_fus <= 0.35 and a_fus <= 1.25 * a_base:      # G4 护栏
            out[k] = dict(M[k])
            out[k]["bbox"] = [round(float(v), 6) for v in fused]
            rules["replaced"] += 1
            repl_log.append((k, round(a_base, 4), round(a_fus, 4),
                             round(iou(fused, m), 3)))
        else:
            out[k] = M[k]
            rules["guard_rejected"] += 1
    else:
        out[k] = M[k]
        rules["kept_baseline"] += 1

json.dump(out, open(f"{OUT}/fusion_g4.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

big_base = [k for k in keys if area(M[k]["bbox"]) > 0.35]
big_g4 = [k for k in keys if area(out[k]["bbox"]) > 0.35]
fixed = len(set(big_base) - set(big_g4))
newbig = len(set(big_g4) - set(big_base))
print("=== G4 结果 ===")
for r, c in rules.most_common():
    print(f"  {r:16s} {c}")
print(f"\n大框: 基线 {len(big_base)} → G4 {len(big_g4)}  (修掉 {fixed}, 新增 {newbig})")
print(f"替换数: {rules['replaced']}  (护栏拦下 {rules['guard_rejected']})")

k = "000247_008"
print(f"\n=== {k} 校验 ===")
print(f"  基线(三模态) bbox={[round(v,3) for v in M[k]['bbox']]}  area={area(M[k]['bbox']):.4f}")
print(f"  L2(RGB)      bbox={[round(v,3) for v in L2[k]['bbox']]}  area={area(L2[k]['bbox']):.4f}")
print(f"  G(RGB)       bbox={[round(v,3) for v in G[k]['bbox']]}  area={area(G[k]['bbox']):.4f}")
print(f"  G4 最终      bbox={[round(v,3) for v in out[k]['bbox']]}  area={area(out[k]['bbox']):.4f}")
print(f"  G4是否替换: {'是' if k in [x[0] for x in repl_log] else '否'}")

print("\n=== 替换后面积放大最多(前10) ===")
for kk, ab, af, iv in sorted(repl_log, key=lambda x: -x[2])[:10]:
    print(f"  {kk}  base_area={ab:.4f} → fused_area={af:.4f}  IoU(fus,M)={iv}")
