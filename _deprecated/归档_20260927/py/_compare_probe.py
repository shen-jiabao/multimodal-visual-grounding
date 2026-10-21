# -*- coding: utf-8 -*-
"""对比探针结果 vs 已锁定基线 result_gd_3modal_fullseq.json，输出决策。

输入：
  probe_result.json                （远端 /srv/output/ 拉回本地）
  result_gd_3modal_fullseq.json    （锁定基线，本地）
  probe/mini_queries.json          （探针问句）
输出：
  probe_comparison.json + 终端汇总（含全量 ETA 与决策建议）
"""
import json
import statistics
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
PROBE = R / "probe_result.json"
BASE = R / "result_gd_3modal_fullseq.json"
QUERIES = R / "probe" / "mini_queries.json"
OUT = R / "probe_comparison.json"

IOU_HI = 0.7      # 高一致
IOU_OK = 0.5      # 中一致（Ordinal 推广共识阈值见项目记忆）
IOU_LO = 0.3


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2 - x1), max(0., y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 1e-9 else 0.0


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def main() -> int:
    if not PROBE.exists():
        print(f"[ERR] 探针结果未拉回本地：{PROBE}")
        return 2
    probe = json.loads(PROBE.read_text(encoding="utf-8"))
    base = json.loads(BASE.read_text(encoding="utf-8"))
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    keys = list(probe)
    print(f"探针 {len(keys)} 问句，基线 {len(base)} 条")

    rows = []
    hits = 0
    for k in keys:
        pq = probe[k]
        boxes = pq.get("boxes") or []
        loc_bbox = boxes[0] if boxes else None
        gd_bbox = base[k]["bbox"]
        query = queries[k]["query"]
        rec = {
            "qid": k,
            "query": query,
            "loc_hit": loc_bbox is not None,
            "loc_n_boxes": len(boxes),
            "loc_bbox": loc_bbox,
            "gd_bbox": gd_bbox,
            "iou": round(iou(loc_bbox, gd_bbox), 4) if loc_bbox else 0.0,
            "time_s": pq.get("time_s"),
        }
        if loc_bbox:
            hits += 1
        rows.append(rec)

    # 汇总
    n = len(rows)
    hit_rate = hits / n * 100 if n else 0.0
    ious = [r["iou"] for r in rows if r["loc_hit"]]
    hi = sum(1 for v in ious if v >= IOU_HI)
    ok = sum(1 for v in ious if v >= IOU_OK)
    hi_excl = [r["qid"] for r in rows
               if r["loc_hit"] and r["iou"] >= IOU_HI and r["loc_n_boxes"] >= 2]
    times = [r["time_s"] for r in rows if r.get("time_s") is not None]
    bad = [r["qid"] for r in rows
           if r["loc_bbox"] and not (0 <= r["loc_bbox"][0] < r["loc_bbox"][2] <= 1
                                     and 0 <= r["loc_bbox"][1] < r["loc_bbox"][3] <= 1)]

    print("\n===== 探针 vs 基线 =====")
    print(f"问句 {n}；Locate 命中(有框) {hits} ({hit_rate:.1f}%)；"
          f"GD 全有框（基线）")
    if ious:
        print(f"IoU: 均值 {statistics.mean(ious):.3f} 中位 "
              f"{statistics.median(ious):.3f}  ≥0.7:{hi} ≥0.5:{ok}")
    else:
        print("IoU: 无命中框可算")
    if bad:
        print(f"⚠️ 越界/非法 bbox {len(bad)}: {bad[:5]}")
    if times:
        avg = statistics.mean(times)
        p50 = statistics.median(times)
        t95 = sorted(times)[int(len(times) * 0.95) - 1]
        print(f"单条耗时: 均值 {avg:.2f}s P50 {p50:.2f}s P95 {t95:.2f}s")
        print(f"全量9555 预估: 均值→{avg*9555/3600:.1f}h  P95→{t95*9555/3600:.1f}h"
              f"  P50→{p50*9555/3600:.1f}h")

    # 差异较大（Locate 与 GD 不一致）条目
    diverge = [r for r in rows if r["loc_hit"] and r["iou"] < IOU_LO]
    print(f"\nLocate与GD分歧(IoU<0.3) {len(diverge)}/{n}:")
    for r in diverge[:8]:
        print(f"  {r['qid']} IoU={r['iou']:.2f} loc={r['loc_bbox']} "
              f"gd={r['gd_bbox']} | {r['query'][:60]}")

    # 决策
    print("\n===== 决策建议 =====")
    if not times:
        print("无耗时数据，无法预估全量，先跑探针取时延")
    elif hit_rate < 80:
        print("保留现有提交（Locate 命中率过低）")
    elif not ious or statistics.median(ious) < IOU_LO:
        print("保留现有提交（与基线不一致，质量风险高）")
    else:
        avg = statistics.mean(times)
        eta_h = avg * 9555 / 3600
        if eta_h <= 6:
            print(f"全量值得做（命中 {hit_rate:.0f}%，IoU中位 "
                  f"{statistics.median(ious):.2f}，ETA≈{eta_h:.1f}h）")
        elif eta_h <= 14:
            print(f"可做但需权衡（命中 {hit_rate:.0f}%，IoU中位 "
                  f"{statistics.median(ious):.2f}，ETA≈{eta_h:.1f}h，建议小样再验）")
        else:
            print(f"不建议全量（ETA≈{eta_h:.1f}h 过长，保留现有提交）")

    json.dump({"rows": rows,
               "summary": {
                   "n": n, "hits": hits, "hit_rate": round(hit_rate, 2),
                   "iou_mean": round(statistics.mean(ious), 4) if ious else None,
                   "iou_median": round(statistics.median(ious), 4) if ious else None,
                   "iou_ge70": hi, "iou_ge50": ok, "bad_bbox": len(bad),
                   "time_avg": round(statistics.mean(times), 3) if times else None,
                   "time_p95": round(sorted(times)[int(len(times)*0.95)-1], 3)
                   if times else None,
                   "eta_h_avg": round(statistics.mean(times)*9555/3600, 2)
                   if times else None,
               }},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n明细已存 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())