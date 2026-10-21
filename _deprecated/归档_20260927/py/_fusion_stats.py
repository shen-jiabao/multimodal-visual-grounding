# -*- coding: utf-8 -*-
"""离线融合统计：对远端已有的 4 份全量预测做一致性分析（纯 CPU）。"""
import json
from pathlib import Path

OUT = Path("/root/output")
FILES = {
    "L1_locate_v1": OUT / "predictions.json",
    "L2_locate_v2_hires": OUT / "predictions_v2_highres_slow.json",
    "G_groundingdino": OUT / "predictions_grounding.json",
    "M_gd3modal": Path("/root/infer/result_gd_3modal_v2.json"),
}


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


data = {}
for name, p in FILES.items():
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        d = {k: v for k, v in d.items() if "bbox" in v}
        data[name] = d
        big = sum(1 for v in d.values() if area(v["bbox"]) > 0.35)
        print(f"{name}: {len(d)} 条, 面积>0.35 的 {big} 条, "
              f"均面积 {sum(area(v['bbox']) for v in d.values())/len(d):.4f}")
    else:
        print(f"{name}: 文件不存在 {p}")

names = list(data.keys())
common = set.intersection(*[set(d) for d in data.values()]) if data else set()
print(f"\n公共 qid: {len(common)}")

print("\n===== 两两 IoU 统计 =====")
pairs = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        na, nb = names[i], names[j]
        ious = [iou(data[na][k]["bbox"], data[nb][k]["bbox"]) for k in common]
        ious.sort()
        n = len(ious)
        mean = sum(ious) / n
        med = ious[n // 2]
        ge50 = sum(1 for x in ious if x >= 0.5) / n
        ge30 = sum(1 for x in ious if x >= 0.3) / n
        lt10 = sum(1 for x in ious if x < 0.1) / n
        pairs[(na, nb)] = ious
        print(f"{na:20s} vs {nb:20s} meanIoU={mean:.3f} med={med:.3f} "
              f"≥0.5:{ge50*100:5.1f}% ≥0.3:{ge30*100:5.1f}% <0.1:{lt10*100:5.1f}%")

print("\n===== 多样本一致性（以最大两两 IoU 计）=====")
buckets = {"≥0.5(可融合)": 0, "0.3~0.5": 0, "0.1~0.3": 0, "<0.1(分歧大)": 0}
detail = {}
for k in common:
    boxes = {n: data[n][k]["bbox"] for n in names}
    best = 0.0; best_pair = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            v = iou(boxes[names[i]], boxes[names[j]])
            if v > best:
                best, best_pair = v, (names[i], names[j])
    if best >= 0.5:
        buckets["≥0.5(可融合)"] += 1
    elif best >= 0.3:
        buckets["0.3~0.5"] += 1
    elif best >= 0.1:
        buckets["0.1~0.3"] += 1
    else:
        buckets["<0.1(分歧大)"] += 1
    detail[k] = {"boxes": boxes, "max_iou": round(best, 4),
                 "best_pair": list(best_pair) if best_pair else None}
print(buckets)
for k, v in buckets.items():
    print(f"  {k}: {v} ({v/len(common)*100:.1f}%)")

out_p = OUT / "fusion_detail.json"
out_p.write_text(json.dumps(detail, ensure_ascii=False), encoding="utf-8")
print(f"\n明细已保存: {out_p} ({out_p.stat().st_size/1024:.0f} KB)")
