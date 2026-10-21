# -*- coding: utf-8 -*-
"""统计并汇总「raw / sort / project」三种链修正结果的结构差异，产出人工复核用的 HTML。

背景：链修正有两种模式，此前离线报告只按「位移小」选了 project。但两者的修法本质不同：

    raw      = 模型原始输出
    sort     = 把链内各框按位置排序后**重新指派**（等价于：把模型的 K 个真实检测
               重新分给 K 个序数 → 输出坐标全部来自模型自己画过的框）
    project  = PAVA 保序回归（保持指派，把框**平移**到最小 L2 单调序列 → 会把
               多个分离的检测**平均到一起**，产出模型从未提出的坐标）

本脚本量化三个可比指标：

  1. **互斥度**：同一条 query 上 IoU(sort, project)。IoU < 0.5 说明两种修法给出的
     框不可能都对 —— 至少有一个是错的。这是"必须人工判读"的规模。
  2. **塌缩**：链内输出框两两 IoU > 0.5 的对数（把多个目标压到同一个位置）。
  3. **新坐标**：输出框中心不在该链任何原始框中心附近（> 0.01）的条数
     （= 该修法造出了模型没给过的位置）。

输出：
    outputs/复赛/chain/chain_modes/index.html   人工复核页（三联图 + 每条 query 的三组坐标）
    outputs/复赛/chain/chain_modes/stats.json   结构化统计

用法：
    python scripts/复赛/chain_modes_report.py \
        --diag outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl \
        --dir outputs/复赛/chain/chain_modes --gap-ratio 0.05
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.chains import chain_of, enforce_ordinal_chains  # noqa: E402
from src.postprocess.search import iou_xyxy  # noqa: E402

TOL = 1e-6
DISAGREE_IOU = 0.5      # sort 与 project 互不满足 ACC@0.5 的门槛
COLLAPSE_IOU = 0.5      # 判定"两条 query 被压到同一个目标"


def load(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def same(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def norm(b):
    return None if b is None else [round(float(v), 6) for v in b]


def center(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--dir", default="outputs/复赛/chain/chain_modes")
    ap.add_argument("--gap-ratio", type=float, default=0.05)
    ap.add_argument("--no-render-check", action="store_true",
                    help="跳过「重算 project 是否等于云端 final_box」的自检")
    args = ap.parse_args()

    recs = load(args.diag)
    by_img = defaultdict(list)
    for r in recs:
        by_img[r["visible"]].append(r)

    stats = {
        "images": len(by_img),
        "chain_images": 0,
        "sort_active_images": 0,
        "project_active_images": 0,
        "mode_diff_images": 0,
        "affected_queries": 0,
        "disagree_queries": 0,          # IoU(sort, project) < 0.5
        "disagree_iou_lt_03": 0,
        "collapse": {"raw": 0, "sort": 0, "project": 0},
        "shift": {"sort": [], "project": []},
        "mismatch_vs_diag_final": 0,
    }
    cases = []

    for vis, rs in by_img.items():
        rs.sort(key=lambda r: r["query_id"])
        qids = [r["query_id"] for r in rs]
        qs = [r["query"] for r in rs]
        raw = [r.get("pre_chain_box") for r in rs]
        diag_final = [r.get("final_box") for r in rs]

        # 链成员：同一 (axis, direction, key) 下 rank 数 >= 2 才算真链
        keys = defaultdict(set)
        for q in qs:
            ch = chain_of(q)
            if ch:
                keys[(ch["axis"], ch["direction"], ch["key"])].add(ch["rank"])
        member = set()
        for i, q in enumerate(qs):
            ch = chain_of(q)
            if ch and len(keys[(ch["axis"], ch["direction"], ch["key"])]) >= 2:
                member.add(i)
        if not member:
            continue
        stats["chain_images"] += 1

        sort_b = enforce_ordinal_chains(qs, raw, mode="sort", gap_ratio=0.0)
        proj_b = enforce_ordinal_chains(qs, raw, mode="project",
                                        gap_ratio=args.gap_ratio)

        if not args.no_render_check:
            for i in range(len(rs)):
                if not same(norm(proj_b[i]), norm(diag_final[i])):
                    stats["mismatch_vs_diag_final"] += 1

        aff = [i for i in range(len(rs))
               if not (same(norm(raw[i]), norm(sort_b[i]))
                       and same(norm(raw[i]), norm(proj_b[i])))]
        if not aff:
            continue
        aff_set = {qids[i] for i in aff}

        s_act = any(not same(norm(raw[i]), norm(sort_b[i])) for i in range(len(rs)))
        p_act = any(not same(norm(raw[i]), norm(proj_b[i])) for i in range(len(rs)))
        stats["sort_active_images"] += int(s_act)
        stats["project_active_images"] += int(p_act)
        mode_diff = any(not same(norm(sort_b[i]), norm(proj_b[i])) for i in range(len(rs)))
        stats["mode_diff_images"] += int(mode_diff)
        stats["affected_queries"] += len(aff)

        # ---- 逐条 query 指标 ----
        detail = []
        for i in aff:
            row = {"i": i + 1, "qid": qids[i], "query": qs[i],
                   "raw": raw[i], "sort": sort_b[i], "project": proj_b[i]}
            if sort_b[i] is not None and proj_b[i] is not None:
                j_iou = iou_xyxy(sort_b[i], proj_b[i])
                row["iou_sort_project"] = j_iou
                if j_iou < DISAGREE_IOU:
                    stats["disagree_queries"] += 1
                if j_iou < 0.3:
                    stats["disagree_iou_lt_03"] += 1
            detail.append(row)

        # ---- 塌缩 / 位移 ----
        for tag, bs in (("raw", raw), ("sort", sort_b), ("project", proj_b)):
            n_coll = 0
            for a in range(len(aff)):
                for b in range(a + 1, len(aff)):
                    ba, bb = bs[aff[a]], bs[aff[b]]
                    if ba is not None and bb is not None and iou_xyxy(ba, bb) > COLLAPSE_IOU:
                        n_coll += 1
            stats["collapse"][tag] += n_coll

        shifts = {"sort": [], "project": []}
        for i in aff:
            if raw[i] is None:
                continue
            c0 = center(raw[i])
            for tag, bs in (("sort", sort_b), ("project", proj_b)):
                if bs[i] is None:
                    continue
                c1 = center(bs[i])
                d = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
                shifts[tag].append(d)
                stats["shift"][tag].append(d)

        cases.append({
            "visible": vis, "file": os.path.basename(vis),
            "order": qids, "queries": qs, "affected": aff_set,
            "detail": detail,
            "n_disagree": sum(1 for d in detail
                              if d.get("iou_sort_project", 1.0) < DISAGREE_IOU),
            "max_sort": max(shifts["sort"]) if shifts["sort"] else 0.0,
            "max_project": max(shifts["project"]) if shifts["project"] else 0.0,
        })

    # ---------------- 打印统计 ----------------
    n_aff = max(1, stats["affected_queries"])
    print("=== 链修正三态对比（raw / sort / project）===")
    print(f"  图像 {stats['images']}；含真链的图 {stats['chain_images']}；"
          f"sort 有改动 {stats['sort_active_images']}；project 有改动 "
          f"{stats['project_active_images']}；两者不同的图 {stats['mode_diff_images']}")
    print(f"  参与修正的 query = {stats['affected_queries']}")
    print(f"  **sort 与 project 互不满足 IoU>=0.5 的 query = {stats['disagree_queries']}"
          f" ({100.0 * stats['disagree_queries'] / n_aff:.1f}%)"
          f"   其中 IoU<0.3 = {stats['disagree_iou_lt_03']}**")
    print(f"  塌缩（链内输出框两两 IoU>0.5 的对数）：raw {stats['collapse']['raw']}  "
          f"sort {stats['collapse']['sort']}  project {stats['collapse']['project']}")
    print("  说明：project 的塌缩来自 PAVA 把多个分离检测平均到同一坐标；"
          "sort 的塌缩只可能来自链内原本重复的框。")
    for tag in ("sort", "project"):
        v = sorted(stats["shift"][tag])
        if v:
            n = len(v)
            print(f"  位移 {tag:8s}: p50={v[n // 2]:.4f} p90={v[min(n - 1, int(n * .9))]:.4f}"
                  f" max={v[-1]:.4f} >0.15 的 {sum(1 for x in v if x > 0.15)}")
    if not args.no_render_check:
        print(f"  自检：重算 project 与云端 diag.final_box 不一致 = "
              f"{stats['mismatch_vs_diag_final']}（应为 0）")

    cases.sort(key=lambda c: (-c["n_disagree"], -c["max_project"]))
    print("\n分歧最大的 12 张图（按 sort/project 互斥条数）：")
    for c in cases[:12]:
        print(f"  {c['file']:14s} 参与 {len(c['affected']):2d}  "
              f"互斥 {c['n_disagree']:2d}  sort最大位移 {c['max_sort']:.3f}  "
              f"project {c['max_project']:.3f}")

    # ---------------- HTML ----------------
    os.makedirs(args.dir, exist_ok=True)
    html_path = os.path.join(args.dir, "index.html")
    parts = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<title>链修正 raw / sort / project 人工复核</title>",
        "<style>body{font:14px/1.6 -apple-system,'Microsoft YaHei',sans-serif;"
        "margin:0;padding:16px;background:#f5f6f8;color:#111}"
        "h1{font-size:20px} h2{font-size:16px;margin:24px 0 8px}"
        "img{max-width:100%;border:1px solid #ccd;background:#fff}"
        "table{border-collapse:collapse;margin:8px 0 0;background:#fff}"
        "th,td{border:1px solid #dde;padding:3px 7px;font-size:12.5px;white-space:nowrap}"
        "th{background:#eef1f6} .dis{background:#ffe9e9} .ok{background:#eaffea}"
        ".q{white-space:normal;max-width:520px} .tag{display:inline-block;padding:1px 7px;"
        "border-radius:9px;font-size:12px;margin-right:6px}"
        ".t-sort{background:#d8f5d8}.t-proj{background:#dbe6ff}"
        ".summary{background:#fff;border:1px solid #dde;padding:10px 14px;border-radius:6px}"
        "code{background:#eef1f6;padding:1px 4px;border-radius:3px}"
        "</style></head><body>",
        "<h1>链修正人工复核：raw（模型原始） / sort（置换重指派） / project（PAVA 平均平移）</h1>",
        "<div class='summary'>",
        "<p>三联图每格都是同一张可见光图：<b>左=raw 中=sort 右=project</b>。"
        "框上的数字 = 该图内 query 的序号（三格一致，对应下表 <code>#</code> 列）。"
        "彩色粗框 = 参与链修正的 query；灰细框 = 该图其余 query（仅作参照）。</p>",
        "<p><b>sort</b> 是纯置换：每条 query 拿到的仍是模型<b>自己画过的某个框</b>"
        "（把 K 个真实检测重新分给 K 个序数）。<br>"
        "<b>project</b> 是保序回归：保持各自指派，把框<b>平移</b>到最小 L2 单调序列，"
        "因此会把分离的检测<b>平均到一起</b>，产出模型没给过的坐标。</p>",
        "</div>",
    ]

    def b2s(b):
        if b is None:
            return "None"
        return "%.3f,%.3f,%.3f,%.3f" % tuple(float(v) for v in b)

    for c in cases:
        parts.append("<h2>%s　<span class='tag t-sort'>sort 最大位移 %.3f</span>"
                     "<span class='tag t-proj'>project 最大位移 %.3f</span>"
                     "<span class='tag'>参与 %d 条 / 互斥 %d 条</span></h2>"
                     % (html.escape(c["file"]), c["max_sort"], c["max_project"],
                        len(c["affected"]), c["n_disagree"]))
        parts.append("<img src='%s' alt='%s'>" % (html.escape(c["file"]),
                                                  html.escape(c["file"])))
        parts.append("<table><tr><th>#</th><th>query_id</th><th class='q'>query</th>"
                     "<th>raw（模型原始）</th><th>sort（置换）</th>"
                     "<th>project（平均平移）</th><th>IoU(sort,project)</th></tr>")
        for d in c["detail"]:
            j = d.get("iou_sort_project")
            cls = "ok" if (j is not None and j >= 0.5) else "dis"
            js = "n/a" if j is None else "%.3f" % j
            parts.append(
                "<tr class='%s'><td>%d</td><td>%s</td><td class='q'>%s</td>"
                "<td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                % (cls, d["i"], html.escape(d["qid"]), html.escape(d["query"]),
                   b2s(d["raw"]), b2s(d["sort"]), b2s(d["project"]), js))
        parts.append("</table>")
    parts.append("</body></html>")
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(f"\n复核页：{html_path}")

    stats_out = dict(stats)
    stats_out["cases"] = [
        {k: (sorted(v) if isinstance(v, set) else v)
         for k, v in c.items() if k != "detail"}
        for c in cases
    ]
    with open(os.path.join(args.dir, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats_out, fh, ensure_ascii=False, indent=2)
    print(f"统计：{os.path.join(args.dir, 'stats.json')}")


if __name__ == "__main__":
    main()