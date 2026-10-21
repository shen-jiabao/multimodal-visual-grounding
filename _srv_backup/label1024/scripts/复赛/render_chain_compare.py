# -*- coding: utf-8 -*-
"""把「链修正前(pre_chain_box)」与「链修正后(final_box)」两组框画在同一张可见光图上，
供人工判读修正是否正确。

约定（图例）：
    红色粗框 + `i`   = 模型原始输出（修正前）
    绿色粗框 + `i'`  = 链修正后的最终框
    黄色箭头         = 同一 query 从「修正前中心」指向「修正后中心」
    蓝色细框 `[qNN]` = 该图**未移动**的其他 query 框（对照用）

输出：
    outputs/复赛/chain/chain_compare/<场景>.png   可视化
    outputs/复赛/chain/chain_compare/legend.txt   编号 → query_id / query / 框坐标 / 位移

用法：
    python scripts/复赛/render_chain_compare.py \
        --diag outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl \
        --images-dir dataset/Images/visible \
        --out-dir outputs/复赛/chain/chain_compare --top 12
    # 只画指定场景：
    ... --only 015282.png 014449.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

TOL = 1e-6
RED = (230, 30, 30)
GREEN = (0, 200, 0)
BLUE = (40, 90, 230)
YELLOW = (255, 200, 0)


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


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= TOL


def center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def dist(a, b):
    ca, cb = center(a), center(b)
    return ((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--images-dir", default="dataset/Images/visible")
    ap.add_argument("--out-dir", default="outputs/复赛/chain/chain_compare")
    ap.add_argument("--top", type=int, default=12, help="画位移最大的 N 张图")
    ap.add_argument("--only", nargs="*", default=None, help="只画这些场景文件名")
    ap.add_argument("--max-width", type=int, default=1500)
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

    # 选出「有移动」的图，按最大位移排序
    cand = []
    for vis, rs in by_img.items():
        moved = [r for r in rs if not same(r.get("pre_chain_box"), r.get("final_box"))]
        if not moved:
            continue
        mx = max((dist(r["pre_chain_box"], r["final_box"]) for r in moved
                  if r.get("pre_chain_box") and r.get("final_box")), default=0.0)
        cand.append((mx, len(moved), vis, moved, rs))
    cand.sort(key=lambda x: -x[0])

    if args.only:
        keep = [c for c in cand if os.path.basename(c[2]) in set(args.only)]
    else:
        keep = cand[:args.top]

    os.makedirs(args.out_dir, exist_ok=True)
    legend_path = os.path.join(args.out_dir, "legend.txt")
    font = None

    with open(legend_path, "w", encoding="utf-8") as lf:
        lf.write("图例：红=修正前(pre_chain_box)  绿=修正后(final_box)  "
                 "蓝=该图未移动的其他 query\n")
        lf.write("=" * 100 + "\n")
        for mx, nmoved, vis, moved, rs in keep:
            base = os.path.basename(vis)
            src = os.path.join(args.images_dir, base)
            if not os.path.exists(src):
                print(f"[skip] 找不到图像 {src}")
                continue
            im = Image.open(src).convert("RGB")
            W, H = im.size
            s = min(1.0, args.max_width / float(W))
            if s < 1.0:
                im = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
            W2, H2 = im.size
            if font is None:
                font = load_font(max(13, int(W2 / 90)))
            draw = ImageDraw.Draw(im)

            def px(b):
                return [float(b[0]) * W2, float(b[1]) * H2,
                        float(b[2]) * W2, float(b[3]) * H2]

            lf.write(f"\n### {base}   移动 {nmoved} 条 / 全图 {len(rs)} 条 query"
                     f"   最大位移 {mx:.3f}\n")

            # 先画未移动的（对照，蓝色细框）
            for r in rs:
                if r in moved:
                    continue
                b = r.get("final_box")
                if not b:
                    continue
                x1, y1, x2, y2 = px(b)
                draw.rectangle([x1, y1, x2, y2], outline=BLUE, width=2)
                draw.text((x1 + 2, max(0, y1 - font.size - 2)),
                          r["query_id"][-3:], fill=BLUE, font=font)

            # 再画移动的：红=前，绿=后
            moved_sorted = sorted(moved, key=lambda r: r["query_id"])
            for i, r in enumerate(moved_sorted, 1):
                pre, fin = r.get("pre_chain_box"), r.get("final_box")
                if pre:
                    x1, y1, x2, y2 = px(pre)
                    draw.rectangle([x1, y1, x2, y2], outline=RED, width=4)
                    draw.text((x1 + 2, max(0, y1 - font.size - 2)),
                              f"{i}", fill=RED, font=font)
                if fin:
                    x1, y1, x2, y2 = px(fin)
                    draw.rectangle([x1, y1, x2, y2], outline=GREEN, width=4)
                    draw.text((x1 + 2, min(H2 - font.size, y2 + 2)),
                              f"{i}'", fill=GREEN, font=font)
                if pre and fin:
                    c0, c1 = center(pre), center(fin)
                    x0, y0 = c0[0] * W2, c0[1] * H2
                    x1, y1 = c1[0] * W2, c1[1] * H2
                    draw.line([x0, y0, x1, y1], fill=YELLOW, width=3)
                    rr = max(3, font.size // 5)
                    for (cx, cy) in ((x0, y0), (x1, y1)):
                        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=YELLOW)

                d = dist(pre, fin) if (pre and fin) else float("nan")
                lf.write(f"  [{i:2d}] {r['query_id']}  位移={d:.3f}\n")
                lf.write(f"        Q     : {r['query']}\n")
                if pre:
                    lf.write(f"        pre   : {[round(v, 4) for v in pre]}\n")
                if fin:
                    lf.write(f"        final : {[round(v, 4) for v in fin]}\n")

            out = os.path.join(args.out_dir, base)
            im.save(out)
            print(f"写出 {out}  （移动 {nmoved} 条，最大位移 {mx:.3f}）")

    print(f"\n图例/明细：{legend_path}")


if __name__ == "__main__":
    main()
