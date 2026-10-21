#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把某个提交件画到图上，产出可视化证据（离线，不需要 GPU / 服务）。

用途
    · 看交付件到底把框画到了哪里
    · 重点看"改动条目"：蓝框 = 基线件，红框 = 本件，一眼可见换成了什么

用法
    python _viz_q38.py --piece result_semi_q38_nonord \
        --base result_semi_n32bf16_nonord --out-dir q38_viz --n 20

输出
    <out-dir>/*.png           单图（带查询原文）
    <out-dir>/_sheet_changed.png    改动条目拼版
    <out-dir>/_sheet_kept.png       未改动条目拼版（对照）
"""
from __future__ import annotations
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

import argparse
import json
import os
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sub_path as SP  # noqa: E402

R = Path(ROOT_S)
DATA = R / "复赛数据集-基于大模型的多模态视觉理解与推理"

BLUE = (60, 130, 255)      # 基线件
RED = (255, 60, 60)        # 本件（已改动）
GREEN = (60, 220, 120)     # 未改动（两件一致）
BG = (24, 24, 24)
WHITE = (255, 255, 255)
GREY = (170, 170, 170)


def _font(size: int) -> ImageFont.ImageFont:
    """取一个可用的 TrueType 字体；失败则退回 PIL 默认位图字体。"""
    for name in ("arial.ttf", "consola.ttf", "segoeui.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except OSError:
                pass
    return ImageFont.load_default()


def _px(b: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    """归一化 bbox → 像素矩形，并保证有序（新版 PIL 不容忍逆序矩形）。"""
    x1, y1, x2, y2 = (b[0] * w, b[1] * h, b[2] * w, b[3] * h)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return int(x1), int(y1), int(x2), int(y2)


def _iou(a: list[float], b: list[float]) -> float:
    """两个归一化框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if (ar + br - inter) > 0 else 0.0


ANNO_W = 640          # 注释图的目标宽度（**先缩放再画字**，字才够大）
FONT_PT = 22          # 说明条字号（缩放后固定，不随原图尺寸变化）
LINE_H = 30           # 行高（必须与字号相称，否则多行文字重叠串行）
WRAP_N = 52           # 查询文本折行宽度（按 640px @ 22pt 估）


def _fit(im: Image.Image) -> Image.Image:
    """把图缩到 ANNO_W 宽（保持长宽比）；已足够小则原样返回。"""
    w, h = im.size
    if w <= ANNO_W:
        return im
    return im.resize((ANNO_W, max(1, int(round(h * ANNO_W / w)))),
                     Image.LANCZOS)


def _annotate(im: Image.Image, base: list[float] | None,
              cur: list[float] | None, lines: list[str]) -> Image.Image:
    """在图上画框（蓝=基线、红=本件），下方加说明条。

    ★ 必须先缩放到 ANNO_W 再画字：原实现按原图宽度定字号（1920 图 → ~50px），
      而说明条行高固定 18px ⇒ 多行查询文本**互相重叠**，肉眼无法辨认。
      改为"先缩放、字号与行高固定且相称"后，单图与拼版都可直接阅读。
    """
    im = _fit(im.convert("RGB")).copy()
    w, h = im.size
    d = ImageDraw.Draw(im)
    lw = max(2, int(w * 0.005))
    if base:
        d.rectangle(_px(base, w, h), outline=BLUE, width=lw)
    if cur:
        d.rectangle(_px(cur, w, h), outline=RED, width=lw)

    f = _font(FONT_PT)
    cap_h = LINE_H * len(lines) + 10
    canvas = Image.new("RGB", (w, h + cap_h), BG)
    canvas.paste(im, (0, cap_h))
    dc = ImageDraw.Draw(canvas)
    for i, t in enumerate(lines):
        dc.text((8, 6 + i * LINE_H), t, fill=WHITE if i == 0 else GREY, font=f)
    return canvas


def _wrap(txt: str, width: int = WRAP_N) -> list[str]:
    return textwrap.wrap(txt, width) or [""]


def _contact_sheet(items: list[tuple[str, Image.Image]], out: str,
                   cols: int = 3) -> None:
    """拼版：**不再缩放**单元格（注释图已是 ANNO_W 宽），否则字又变小。

    原先 4 列 × 340px 会把 22px 的字压到 ~8px ⇒ 与单图一样看不清。
    """
    if not items:
        return
    cw = max(it.width for _, it in items)
    ch = max(it.height for _, it in items)
    pad = 28
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cw, rows * (ch + pad)), BG)
    d = ImageDraw.Draw(sheet)
    f = _font(18)
    for i, (tag, im) in enumerate(items):
        r, c = divmod(i, cols)
        y = r * (ch + pad)
        sheet.paste(im, (c * cw + (cw - im.width) // 2, y + pad))
        d.text((c * cw + 8, y + 5), tag, fill=WHITE, font=f)
    sheet.save(out)


def run_images(q: dict, cur: dict, base: dict, out_dir: str,
               n_img: int) -> int:
    """按图名排序，取前 n_img 张图；每张图画出它名下**所有**查询的框。

    颜色：红 = 本件；蓝 = 基线（仅在该条改动时画）；绿 = 未改动（两件一致）。
    """
    # 按图名分组（qid 形如 000001_003，图名即前缀）
    groups: dict[str, list[str]] = {}
    for k in sorted(cur):
        groups.setdefault(k.split("_")[0], []).append(k)
    names = sorted(groups)[:n_img]
    if not names:
        print("没有可画的图")
        return 1

    odir = Path(out_dir)
    odir.mkdir(parents=True, exist_ok=True)
    items, n_ch_all, n_q_all = [], 0, 0

    for nm in names:
        k0 = groups[nm][0]
        src = DATA / q[k0]["visible"]
        if not src.exists():
            print(f"  ✗ 缺图 {src}")
            continue
        im = _fit(Image.open(src).convert("RGB"))
        w, h = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, int(w * 0.005))
        caps = []
        for k in groups[nm]:
            c0 = cur[k]["bbox"]
            b0 = (base.get(k) or {}).get("bbox")
            if not c0:
                caps.append(f"{k}  [本件无框]")
                continue
            ch = bool(b0) and c0 != b0
            n_q_all += 1
            if ch:
                n_ch_all += 1
                d.rectangle(_px(b0, w, h), outline=BLUE, width=lw)
            d.rectangle(_px(c0, w, h),
                        outline=RED if ch else GREEN, width=lw)
            caps.append(f"{k} {'改' if ch else '同'}  Q: {q[k]['query']}")
        lines = [f"{nm}  共 {len(groups[nm])} 条"] + [
            t for cap in caps for t in _wrap(cap)]
        ann = _annotate(im, None, None, lines[:14])
        ann.save(odir / f"img_{nm}.png")
        items.append((f"{nm}  ({sum(1 for x in caps if ' 改 ' in x)}改"
                      f"/{len(caps)})", ann))
        print(f"  {nm}  条 {len(groups[nm])}  "
              f"改 {sum(1 for x in caps if ' 改 ' in x)}")

    # 每 20 张一个拼版
    for i in range(0, len(items), 20):
        chunk = items[i:i + 20]
        _contact_sheet(chunk, str(odir /
                                 f"_sheet_img_{i // 20 + 1:02d}.png"))
    print(f"\n图 {len(names)} 张｜查询 {n_q_all} 条｜其中改动 {n_ch_all} 条")
    print(f"  红=本件 蓝=基线 绿=未改动｜拼版 {odir}/_sheet_img_*.png")
    print("=== DONE ===")
    return 0


def _resolve(spec: str):
    """先按路径解析（存在即用），否则按 `_sub_path` 名解析。"""
    p = Path(spec)
    if p.exists():
        return p
    return SP.find(spec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", default="result_semi_q38_nonord",
                    help="本件（路径或 _sub_path 名）")
    ap.add_argument("--base", default="result_semi_n32bf16_nonord",
                    help="基线件（蓝框；留空则不画）")
    ap.add_argument("--out-dir", default=str(R / "q38_viz"))
    ap.add_argument("--mode", choices=["entries", "images"],
                    default="entries",
                    help="entries=按条目抽样；images=按图名前 N 张")
    ap.add_argument("--n", type=int, default=20,
                    help="entries 模式=每类条数；images 模式=图张数")
    ap.add_argument("--seed", type=int, default=20261003)
    a = ap.parse_args()

    q = json.loads((DATA / "queries" / "queries.json").read_text("utf-8"))
    p_piece = _resolve(a.piece)
    cur = json.loads(p_piece.read_text("utf-8"))
    base = json.loads(_resolve(a.base).read_text("utf-8")) if a.base else {}
    print(f"本件 {p_piece.name}｜条目 {len(cur)}")

    if a.mode == "images":
        return run_images(q, cur, base, a.out_dir, a.n)

    # 分类：改动 / 未改动（以 bbox 是否逐位相同为准）
    changed, kept = [], []
    for k, v in cur.items():
        b = (base.get(k) or {}).get("bbox")
        c = v.get("bbox")
        if not c or not b:
            continue
        (changed if c != b else kept).append(k)

    # 改动条目按 IoU 分层抽样（跨越"完全不同 → 临界"整个区间）
    ious = {k: _iou(base[k]["bbox"], cur[k]["bbox"]) for k in changed}
    changed.sort(key=lambda k: ious[k])
    n = min(a.n, len(changed))
    step = max(1, len(changed) // n)
    pick_ch = [changed[i] for i in range(0, len(changed), step)][:n]

    random.seed(a.seed)
    pick_kept = random.sample(kept, min(a.n, len(kept)))

    odir = Path(a.out_dir)
    odir.mkdir(parents=True, exist_ok=True)

    for tag, pick in (("changed", pick_ch), ("kept", pick_kept)):
        items = []
        for k in pick:
            src = DATA / q[k]["visible"]
            if not src.exists():
                print(f"  ✗ 缺图 {src}")
                continue
            im = Image.open(src)
            b0 = (base.get(k) or {}).get("bbox")
            c0 = cur[k]["bbox"]
            io = _iou(b0, c0) if b0 else 0.0
            lines = [f"{k}   IoU(蓝,红)={io:.3f}", *[
                f"Q: {t}" for t in _wrap(q[k]["query"])]]
            ann = _annotate(im, b0, c0, lines)
            ann.save(odir / f"{tag}_{k}.png")
            items.append((f"{k}  IoU={io:.2f}", ann))
        _contact_sheet(items, str(odir / f"_sheet_{tag}.png"))

    print(f"\n改动条目 {len(changed)}｜未改动 {len(kept)}")
    if changed:
        vs = sorted(ious.values())
        print(f"  改动面 IoU：p10={vs[len(vs)//10]:.3f} "
              f"p50={vs[len(vs)//2]:.3f} p90={vs[9*len(vs)//10]:.3f}")

    # 抽到的改动条目明细（便于与图逐条对照）
    print("\n改动条目抽样明细（蓝=基线，红=本件）：")
    for k in pick_ch:
        b0 = (base.get(k) or {}).get("bbox") or []
        c0 = cur[k]["bbox"]
        bs = ", ".join(f"{v:.3f}" for v in b0) if b0 else "—"
        cs = ", ".join(f"{v:.3f}" for v in c0)
        print(f"  {k}  IoU={ious[k]:.3f}")
        print(f"     基线 [{bs}]")
        print(f"     本件 [{cs}]")
        print(f"     Q: {q[k]['query']}")

    print(f"\n  图 {odir}")
    print(f"    {odir / '_sheet_changed.png'}   （蓝=基线，红=本件）")
    print(f"    {odir / '_sheet_kept.png'}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
