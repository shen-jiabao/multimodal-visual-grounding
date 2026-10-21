# -*- coding: utf-8 -*-
"""生成链修正人工复核页（自包含 HTML，图片用相对路径引用）。

每张场景一块，含：

    1. overlay/<场景>.png      三色同图：红=raw 绿=sort 蓝=project（+ 位移连线）
    2. montage/<场景>.png      三联缩略：左 RAW / 中 SORT / 右 PROJECT
    3. 表格                    该图每条参与修正的 query 的三组坐标 + IoU(sort,project)
    4. 三个单格大图链接         panels/<场景>_{raw,sort,project}.png

用法：
    python scripts/复赛/chain_modes_index.py --dir outputs/复赛/chain/chain_modes
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.chains import (  # noqa: E402
    chain_of,
    classify_chain,
    enforce_ordinal_chains,
    residual_tokens,
)
from src.postprocess.search import iou_xyxy  # noqa: E402

TOL = 1e-6
_ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
        6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th"}
_STATUS_CN = {
    "ordered": "已单调（本来就对）",
    "A": "破序·无解释",
    "B_dup": "破序·框重复",
    "B_tie": "破序·并列",
    "B_broken": "破序·断裂",
    "skip_rank": "序数不连续",
    "skip_axis": "轴向不一致",
    "has_none": "含缺失框",
}


def short_label(q):
    ch = chain_of(q)
    rt = residual_tokens(q)
    parts = []
    if ch:
        parts.append(_ORD.get(ch["rank"], "#%d" % ch["rank"]))
    if rt:
        parts.append(max(rt, key=len))
    return " ".join(parts)


def norm(b):
    return None if b is None else [round(float(v), 6) for v in b]


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= TOL


def b2s(b):
    if b is None:
        return "None"
    return "%.3f, %.3f, %.3f, %.3f" % tuple(float(v) for v in b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--dir", default="outputs/复赛/chain/chain_modes")
    ap.add_argument("--gap-ratio", type=float, default=0.05)
    args = ap.parse_args()

    by_img = defaultdict(list)
    with open(args.diag, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_img[r["visible"]].append(r)

    scenes = []
    for vis, rs in by_img.items():
        rs.sort(key=lambda r: r["query_id"])
        qids = [r["query_id"] for r in rs]
        qs = [r["query"] for r in rs]
        raw = [r.get("pre_chain_box") for r in rs]

        groups = defaultdict(list)
        for i, q in enumerate(qs):
            ch = chain_of(q)
            if ch:
                groups[(ch["axis"], ch["direction"], ch["key"])].append(i)
        real = {k: v for k, v in groups.items()
                if len({chain_of(qs[i])["rank"] for i in v}) >= 2}
        if not real:
            continue

        # 每条 query 的链状态
        st_map = {}
        for k, idxs in real.items():
            if any(raw[i] is None for i in idxs):
                st = "has_none"
            else:
                items = []
                for i in idxs:
                    ch = chain_of(qs[i])
                    items.append((qids[i], qs[i], raw[i],
                                  {"axis": ch["axis"], "direction": ch["direction"],
                                   "rank": ch["rank"]}))
                st = classify_chain(items)
            for i in idxs:
                st_map[i] = st

        sort_b = enforce_ordinal_chains(qs, raw, mode="sort", gap_ratio=0.0)
        proj_b = enforce_ordinal_chains(qs, raw, mode="project",
                                        gap_ratio=args.gap_ratio)
        aff = [i for i in range(len(rs))
               if not (same(norm(raw[i]), norm(sort_b[i]))
                       and same(norm(raw[i]), norm(proj_b[i])))]
        if not aff:
            continue

        rows = []
        for i in aff:
            j = None
            if sort_b[i] is not None and proj_b[i] is not None:
                j = iou_xyxy(sort_b[i], proj_b[i])
            rows.append({
                "qid": qids[i], "query": qs[i], "st": st_map.get(i, "?"),
                "label": short_label(qs[i]),
                "raw": raw[i], "sort": sort_b[i], "project": proj_b[i],
                "iou": j,
            })
        # 按链状态（破序优先）再按互斥程度排
        rows.sort(key=lambda r: (r["st"] == "ordered",
                                 -(r["iou"] if r["iou"] is not None else 1.0)))
        scenes.append({"file": os.path.basename(vis), "rows": rows})

    # 破序条数多的排前面
    scenes.sort(key=lambda s: (-sum(1 for r in s["rows"] if r["st"] != "ordered"),
                               s["file"]))

    n_rows = sum(len(s["rows"]) for s in scenes)
    n_dis = sum(1 for s in scenes for r in s["rows"]
                if r["iou"] is not None and r["iou"] < 0.5)
    n_broken = sum(1 for s in scenes for r in s["rows"] if r["st"] != "ordered")

    parts = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>链修正人工复核 · raw / sort / project</title>",
        "<style>",
        "body{font:14px/1.65 -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;",
        "margin:0;padding:18px 22px;background:#f4f5f7;color:#16181d}",
        "h1{font-size:21px;margin:0 0 6px} h2{font-size:17px;margin:30px 0 10px;",
        "padding:8px 12px;background:#fff;border-left:4px solid #2f6fed;border-radius:4px}",
        ".lead{background:#fff;border:1px solid #e2e5ea;border-radius:8px;",
        "padding:14px 18px;margin-bottom:8px}",
        ".lead b{color:#c02020}",
        "img{max-width:100%;display:block;border:1px solid #d5d9e0;",
        "border-radius:5px;background:#fff;margin:6px 0}",
        "table{border-collapse:collapse;margin:10px 0 4px;background:#fff;",
        "width:100%;font-size:12.5px}",
        "th,td{border:1px solid #e0e3e9;padding:5px 8px;vertical-align:top}",
        "th{background:#eef1f6;font-weight:600;white-space:nowrap}",
        "td.q{max-width:360px}",
        "tr.dis td{background:#fff0f0} tr.dis td.iou{color:#c02020;font-weight:700}",
        "tr.ok td.iou{color:#1a7a1a}",
        ".st{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11.5px;",
        "white-space:nowrap} .st-bad{background:#ffdede;color:#a01010}",
        ".st-ok{background:#e2f5e2;color:#186818}",
        ".links a{margin-right:14px;font-size:12.5px}",
        "code{background:#eef1f6;padding:1px 5px;border-radius:3px}",
        ".cap{color:#5b6270;font-size:12.5px;margin:2px 0 0}",
        "</style></head><body>",
        "<h1>链修正人工复核：raw / sort / project</h1>",
        "<div class='lead'>",
        "<p><b>怎么读：</b>每张图先看 <b>overlay</b>（三色同图）——"
        "<span style='color:#eb1e1e;font-weight:700'>红 = raw</span>（模型原始）、"
        "<span style='color:#00b400;font-weight:700'>绿 = sort</span>（置换重指派）、"
        "<span style='color:#1e5aeb;font-weight:700'>蓝 = project</span>（PAVA 平均平移）。"
        "框上写 <code>序号+模式字母</code>（如 <code>2R</code>=第2条 query 的 raw 框，"
        "<code>2S</code>=它的 sort 框）。细线从 raw 中心指向 sort / project 中心，"
        "一眼看出被挪到哪。</p>",
        "<p><b>判据只有一句：</b>raw 破序是因为「<b>序数贴错</b>」还是「<b>框画偏</b>」？"
        "若各框都盖在真实目标上、只是序数标签反了 → <b>sort 对</b>（它把框重新分给序数，"
        "坐标全是模型自己画过的）；若框各自目标没错、只是歪了一点 → "
        "<b>project 的小幅修正更稳</b>（sort 的置换会把框甩到别的目标上）。</p>",
        "<p><b>两个硬信号：</b>① <code>project</code> 会把分离的检测<b>平均到一起</b>"
        "（统计上造成 32 对框 IoU&gt;0.5 的重合，sort 只有 5 对）——"
        "若两个目标真实分离，重合的那对必然至少一个错；"
        "② 链修正<b>只沿约束轴调整</b>（本数据集 192 组 x 轴、4 组 y 轴），"
        "垂直方向坐标原样保留，所以要看垂直方向是否仍落在目标上。</p>",
        "<p class='cap'>共 <b>%d</b> 张场景、<b>%d</b> 条参与修正的 query；"
        "其中来自「raw 已破序」的 <b>%d</b> 条，"
        "<b>%d</b> 条上两种修法互不满足 IoU≥0.5（表格红底行）。</p>"
        % (len(scenes), n_rows, n_broken, n_dis),
        "</div>",
    ]

    for s in scenes:
        f = s["file"]
        stem = os.path.splitext(f)[0]
        n_bad = sum(1 for r in s["rows"] if r["st"] != "ordered")
        n_d = sum(1 for r in s["rows"] if r["iou"] is not None and r["iou"] < 0.5)
        parts.append("<h2>%s　<span class='st %s'>破序 %d 条</span>　"
                     "<span class='st %s'>两法互斥 %d 条</span></h2>"
                     % (html.escape(f), "st-bad" if n_bad else "st-ok", n_bad,
                        "st-bad" if n_d else "st-ok", n_d))
        parts.append("<div class='links'>大图："
                     "<a href='panels/%s_raw.png' target='_blank'>RAW</a>"
                     "<a href='panels/%s_sort.png' target='_blank'>SORT</a>"
                     "<a href='panels/%s_project.png' target='_blank'>PROJECT</a>"
                     "</div>" % (stem, stem, stem))
        parts.append("<img src='overlay/%s' alt='%s overlay'>" % (html.escape(f), f))
        parts.append("<img src='montage/%s' alt='%s montage'>" % (html.escape(f), f))

        parts.append("<table><tr><th>#</th><th>query_id</th><th class='q'>query</th>"
                     "<th>raw 链状态</th><th>raw（模型原始）</th>"
                     "<th>sort（置换）</th><th>project（平均平移）</th>"
                     "<th>IoU(sort,project)</th></tr>")
        for r in s["rows"]:
            iou = r["iou"]
            cls = "dis" if (iou is not None and iou < 0.5) else "ok"
            st_cls = "st-bad" if r["st"] != "ordered" else "st-ok"
            parts.append(
                "<tr class='%s'><td>%s</td><td>%s</td><td class='q'>%s</td>"
                "<td><span class='st %s'>%s</span></td>"
                "<td>%s</td><td>%s</td><td>%s</td><td class='iou'>%s</td></tr>"
                % (cls, html.escape(r["label"] or "-"), html.escape(r["qid"]),
                   html.escape(r["query"]), st_cls,
                   html.escape(_STATUS_CN.get(r["st"], r["st"])),
                   b2s(r["raw"]), b2s(r["sort"]), b2s(r["project"]),
                   "n/a" if iou is None else "%.3f" % iou))
        parts.append("</table>")

    parts.append("</body></html>")
    out = os.path.join(args.dir, "index.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print("复核页：%s" % out)
    print("场景 %d；参与修正 query %d；破序 %d；两法互斥 %d"
          % (len(scenes), n_rows, n_broken, n_dis))


if __name__ == "__main__":
    main()