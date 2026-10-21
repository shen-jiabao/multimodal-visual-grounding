# -*- coding: utf-8 -*-
"""把「模型原始框 / sort 修正 / project 修正」画到可见光图上，供人工判读。

背景：链修正有两种模式，此前离线报告只按「位移小」选了 project，但两者的
**修法本质完全不同**：

    raw      = 模型原始输出（可能把序数指派错）
    sort     = 把链内各框按位置排序后**重新指派**（纯置换，坐标全部来自模型画过的框）
    project  = PAVA 保序回归（保持指派，把框**平移**到最小 L2 单调序列，会把
               分离的检测平均到一起，产出模型从未提出的坐标）

产出三套图，配合使用：

    montage/<场景>.png         三联缩略总览（左 RAW / 中 SORT / 右 PROJECT），快速扫
    panels/<场景>_<模式>.png    单模式**原始分辨率**大图，看细节用
    overlay/<场景>.png         三色叠加（红=raw 绿=sort 蓝=project 同图），看分歧用
    legend.txt                 逐条 raw/sort/project 坐标 + 标签对照

图上约定：
    彩色粗框 = 参与链修正的 query，框角标「序号 序数 物体」（如 `2 2nd kite`）
    灰色细框 = 该图其余 query（不参与修正，仅作位置参照）
    序号 = 该图内 query 按 query_id 升序，三格/三色一致，对应 legend.txt 的 [i]

用法：
    python scripts/复赛/render_chain_modes.py \
        --diag outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl \
        --images-dir dataset/Images/visible \
        --out-dir outputs/复赛/chain/chain_modes
    # 只看指定场景：
    ... --only 015282.png 012594.png
    # 三联总览的每格宽度（0=原始分辨率，文件会很大）：
    ... --montage-width 900
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.inference.chains import (  # noqa: E402
    chain_of,
    enforce_ordinal_chains,
    residual_tokens,
)

TOL = 1e-6
RED = (235, 30, 30)
GREEN = (0, 180, 0)
BLUE = (30, 90, 235)
GRAY = (140, 140, 140)
BLACK = (0, 0, 0)

MODE_COLORS = {"raw": RED, "sort": GREEN, "project": BLUE}
MODE_TITLES = {
    "raw": "RAW = model output (before chain fix)",
    "sort": "SORT = permute: reuse the model's own boxes, reassign to ordinals",
    "project": "PROJECT = PAVA: keep assignment, shift boxes to be monotone",
}

_ORD_ABBR = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
             6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th"}


def load_font(size):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        for root in (r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(root, name)
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    pass
    return ImageFont.load_default()


def short_label(query: str) -> str:
    """图上短标签，如 "2nd kite"（序数缩写 + 最长的实义词）。"""
    ch = chain_of(query)
    rt = residual_tokens(query)
    parts = []
    if ch:
        parts.append(_ORD_ABBR.get(ch["rank"], "#%d" % ch["rank"]))
    if rt:
        parts.append(max(rt, key=len))
    return " ".join(parts) if parts else query[:16]


def same(a, b, tol=TOL):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def norm(b):
    if b is None:
        return None
    return [round(float(v), 6) for v in b]


def text_outlined(dr, xy, text, fill, font):
    """带黑描边的文字：任何背景上都可读。"""
    x, y = xy
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                dr.text((x + dx, y + dy), text, fill=BLACK, font=font)
    dr.text((x, y), text, fill=fill, font=font)


def _px(b, W, H):
    return [float(b[0]) * W, float(b[1]) * H, float(b[2]) * W, float(b[3]) * H]


def _resize_to(im, width):
    if not width or width <= 0 or im.size[0] == width:
        return im
    return im.resize((width, int(im.size[1] * width / im.size[0])), Image.LANCZOS)


def draw_panel(base_img, box_map, order, affected, mode, font, width, labels):
    """单模式面板：受影响 query 用该模式的粗彩框 + 标签，其余灰细框。"""
    im = base_img.copy()
    W, H = im.size
    dr = ImageDraw.Draw(im)

    for qid in order:
        if qid in affected:
            continue
        b = box_map.get(qid)
        if not b:
            continue
        dr.rectangle(_px(b, W, H), outline=GRAY, width=1)

    col = MODE_COLORS[mode]
    for i, qid in enumerate(order, 1):
        if qid not in affected:
            continue
        b = box_map.get(qid)
        if not b:
            text_outlined(dr, (8, 8 + (i - 1) * (font.size + 4)),
                          "%d: none" % i, col, font)
            continue
        x1, y1, x2, y2 = _px(b, W, H)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=4)
        txt = ("%d %s" % (i, labels.get(qid, ""))).strip()
        ty = y1 - font.size - 5
        if ty < 0:
            ty = y1 + 3
        text_outlined(dr, (x1 + 2, ty), txt, col, font)

    out = _resize_to(im, width)
    # 底部信息带（缩放后画，保证字号）
    bar = Image.new("RGB", (out.size[0], font.size + 10), (0, 0, 0))
    ImageDraw.Draw(bar).text((8, 5), "%s   (n=%d)" % (MODE_TITLES[mode], len(order)),
                             fill=(255, 255, 255), font=font)
    canvas = Image.new("RGB", (out.size[0], out.size[1] + bar.size[1]), (0, 0, 0))
    canvas.paste(bar, (0, 0))
    canvas.paste(out, (0, bar.size[1]))
    return canvas


def draw_overlay(base_img, box_map, order, affected, font, width, labels):
    """三色叠加：同一张图上画 raw(红) / sort(绿) / project(蓝)，

    标签形如 `2R` / `2S` / `2P`，并从 raw 中心向 sort / project 中心画细线，
    直接显示两者把框挪到了哪。
    """
    im = base_img.copy()
    W, H = im.size
    dr = ImageDraw.Draw(im)

    # 无关 query：灰细框
    for qid in order:
        if qid in affected:
            continue
        b = box_map.get("raw", {}).get(qid)
        if not b:
            continue
        dr.rectangle(_px(b, W, H), outline=GRAY, width=1)

    def cx(b):
        x1, y1, x2, y2 = _px(b, W, H)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    for i, qid in enumerate(order, 1):
        if qid not in affected:
            continue
        base = box_map["raw"].get(qid)
        for mode, suffix in (("raw", "R"), ("sort", "S"), ("project", "P")):
            b = box_map[mode].get(qid)
            if not b:
                continue
            x1, y1, x2, y2 = _px(b, W, H)
            col = MODE_COLORS[mode]
            dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
            txt = "%d%s" % (i, suffix)
            text_outlined(dr, (x1 + 2, y1 + 2), txt, col, font)
            # 从 raw 中心到本模式中心的连线（除 raw 自身）
            if base is not None and mode != "raw":
                c0, c1 = cx(base), cx(b)
                dr.line([c0[0], c0[1], c1[0], c1[1]], fill=col, width=2)
                rr = max(3, font.size // 6)
                dr.ellipse([c1[0] - rr, c1[1] - rr, c1[0] + rr, c1[1] + rr], fill=col)

    out = _resize_to(im, width)
    bar = Image.new("RGB", (out.size[0], font.size + 10), (0, 0, 0))
    ImageDraw.Draw(bar).text(
        (8, 5), "OVERLAY: RED=raw  GREEN=sort  BLUE=project   (n=%d)" % len(order),
        fill=(255, 255, 255), font=font)
    canvas = Image.new("RGB", (out.size[0], out.size[1] + bar.size[1]), (0, 0, 0))
    canvas.paste(bar, (0, 0))
    canvas.paste(out, (0, bar.size[1]))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--images-dir", default="dataset/Images/visible")
    ap.add_argument("--out-dir", default="outputs/复赛/chain/chain_modes")
    ap.add_argument("--gap-ratio", type=float, default=0.05)
    ap.add_argument("--montage-width", type=int, default=900,
                    help="三联总览每格宽度；0=原始分辨率")
    ap.add_argument("--panel-width", type=int, default=0,
                    help="单格大图宽度；0=原始分辨率（默认，看细节）")
    ap.add_argument("--overlay-width", type=int, default=0,
                    help="叠加图宽度；0=原始分辨率（默认）")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--no-panels", action="store_true")
    ap.add_argument("--no-overlay", action="store_true")
    args = ap.parse_args()

    recs = []
    with open(args.diag, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    by_img = defaultdict(list)
    for r in recs:
        by_img[r["visible"]].append(r)

    n_chain_img = n_sort_act = n_proj_act = n_mode_diff = 0
    collapse = {"raw": 0, "sort": 0, "project": 0}
    proj_mismatch = 0
    cases = []

    for vis, rs in by_img.items():
        rs.sort(key=lambda r: r["query_id"])
        qids = [r["query_id"] for r in rs]
        qs = [r["query"] for r in rs]
        raw = [r.get("pre_chain_box") for r in rs]
        diag_final = [r.get("final_box") for r in rs]

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
        n_chain_img += 1

        sort_b = enforce_ordinal_chains(qs, raw, mode="sort", gap_ratio=0.0)
        proj_b = enforce_ordinal_chains(qs, raw, mode="project",
                                        gap_ratio=args.gap_ratio)

        for i in range(len(rs)):
            if not same(norm(proj_b[i]), norm(diag_final[i])):
                proj_mismatch += 1

        aff = [i for i in range(len(rs))
               if not (same(norm(raw[i]), norm(sort_b[i]))
                       and same(norm(raw[i]), norm(proj_b[i])))]
        if not aff:
            continue
        affected_qids = {qids[i] for i in aff}

        n_sort_act += int(any(not same(norm(raw[i]), norm(sort_b[i]))
                              for i in range(len(rs))))
        n_proj_act += int(any(not same(norm(raw[i]), norm(proj_b[i]))
                              for i in range(len(rs))))
        mode_diff = any(not same(norm(sort_b[i]), norm(proj_b[i]))
                        for i in range(len(rs)))
        n_mode_diff += int(mode_diff)

        for tag, bs in (("raw", raw), ("sort", sort_b), ("project", proj_b)):
            cnt = defaultdict(int)
            for i in aff:
                if bs[i] is not None:
                    cnt[tuple(norm(bs[i]))] += 1
            collapse[tag] += sum(c - 1 for c in cnt.values() if c > 1)

        maxshift = {"sort": 0.0, "project": 0.0}
        for i in aff:
            if raw[i] is None:
                continue
            c0 = ((raw[i][0] + raw[i][2]) / 2, (raw[i][1] + raw[i][3]) / 2)
            for tag, bs in (("sort", sort_b), ("project", proj_b)):
                if bs[i] is None:
                    continue
                c1 = ((bs[i][0] + bs[i][2]) / 2, (bs[i][1] + bs[i][3]) / 2)
                d = ((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2) ** 0.5
                maxshift[tag] = max(maxshift[tag], d)

        cases.append({
            "visible": vis, "file": os.path.basename(vis),
            "qids": qids, "queries": qs, "raw": raw,
            "sort": sort_b, "project": proj_b,
            "affected": affected_qids, "mode_diff": mode_diff,
            "maxshift": maxshift,
        })

    print("图像 %d；含多 rank 链的图 %d" % (len(by_img), n_chain_img))
    print("sort 有改动 %d 图；project 有改动 %d 图；两者结果不同 %d 图 ← 需要人工判读"
          % (n_sort_act, n_proj_act, n_mode_diff))
    print("自检：重算 project 与云端 diag.final_box 不一致 = %d（应为 0）" % proj_mismatch)
    print("塌缩（链内输出框两两 IoU>0.5 的对数）：raw %d  sort %d  project %d"
          % (collapse["raw"], collapse["sort"], collapse["project"]))

    cases.sort(key=lambda c: (-c["maxshift"]["project"], c["visible"]))
    if args.only:
        keep = [c for c in cases if c["file"] in set(args.only)]
    else:
        keep = [c for c in cases if c["mode_diff"]]
    if args.max_images:
        keep = keep[:args.max_images]

    os.makedirs(args.out_dir, exist_ok=True)
    dirs = {}
    for name, flag in (("montage", False), ("panels", args.no_panels),
                       ("overlay", args.no_overlay)):
        if not flag:
            dirs[name] = os.path.join(args.out_dir, name)
            os.makedirs(dirs[name], exist_ok=True)

    legend = os.path.join(args.out_dir, "legend.txt")
    font = None
    n_done = 0

    with open(legend, "w", encoding="utf-8") as lf:
        lf.write("三套图配合看：\n")
        lf.write("  montage/<场景>.png       三联缩略（左 RAW / 中 SORT / 右 PROJECT），快速扫\n")
        lf.write("  panels/<场景>_<模式>.png 单模式原始分辨率大图，看细节\n")
        lf.write("  overlay/<场景>.png       三色同图（红=raw 绿=sort 蓝=project）+ 位移连线\n")
        lf.write("彩色粗框 = 参与链修正的 query，框角标「序号 序数 物体」；"
                 "灰细框 = 该图其余 query（仅参照）\n")
        lf.write("=" * 100 + "\n")

        for c in keep:
            base = c["file"]
            src = os.path.join(args.images_dir, base)
            if not os.path.exists(src):
                print("[skip] 缺少图像 %s" % src)
                continue
            base_img = Image.open(src).convert("RGB")
            if font is None:
                font = load_font(max(18, base_img.size[0] // 52))

            order = c["qids"]
            labels = {q: short_label(c["queries"][i])
                      for i, q in enumerate(order)}
            box_sets = {"raw": c["raw"], "sort": c["sort"],
                        "project": c["project"]}

            # 1) 单模式原始分辨率大图
            if "panels" in dirs:
                stem = os.path.splitext(base)[0]
                for mode in ("raw", "sort", "project"):
                    bm = {q: box_sets[mode][i] for i, q in enumerate(order)}
                    p = draw_panel(base_img, bm, order, c["affected"],
                                   mode, font, args.panel_width, labels)
                    p.save(os.path.join(dirs["panels"],
                                        "%s_%s.png" % (stem, mode)))

            # 2) 三色叠加（原始分辨率）
            if "overlay" in dirs:
                bm = {mode: {q: box_sets[mode][i] for i, q in enumerate(order)}
                      for mode in box_sets}
                ov = draw_overlay(base_img, bm, order, c["affected"],
                                  font, args.overlay_width, labels)
                ov.save(os.path.join(dirs["overlay"], base))

            # 3) 三联缩略总览
            if "montage" in dirs:
                panels = []
                for mode in ("raw", "sort", "project"):
                    bm = {q: box_sets[mode][i] for i, q in enumerate(order)}
                    panels.append(draw_panel(base_img, bm, order, c["affected"],
                                             mode, font, args.montage_width,
                                             labels))
                h = max(p.size[1] for p in panels)
                W = sum(p.size[0] for p in panels) + 2 * 8
                canvas = Image.new("RGB", (W, h), (25, 25, 25))
                x = 0
                for p in panels:
                    canvas.paste(p, (x, 0))
                    x += p.size[0] + 8
                canvas.save(os.path.join(dirs["montage"], base))

            n_done += 1
            print("写出 %s  （链内 %d 条；sort 最大位移 %.3f，project %.3f）"
                  % (base, len(c["affected"]), c["maxshift"]["sort"],
                     c["maxshift"]["project"]))

            lf.write("\n### %s   参与修正 %d 条\n" % (base, len(c["affected"])))
            for i, qid in enumerate(order, 1):
                if qid not in c["affected"]:
                    continue
                j = order.index(qid)
                lf.write("  [%2d] %s    %s\n" % (i, qid, labels.get(qid, "")))
                lf.write("        Q       : %s\n" % c["queries"][j])
                for tag in ("raw", "sort", "project"):
                    b = box_sets[tag][j]
                    lf.write("        %-8s: %s\n"
                             % (tag, "None" if b is None
                                else [round(v, 4) for v in b]))

    print("\n明细：%s" % legend)
    for name, d in dirs.items():
        print("%s：%s" % (name, d))
    print("共渲染 %d 张场景" % n_done)


if __name__ == "__main__":
    main()
