# -*- coding: utf-8 -*-
"""问题条目复核图：带图例 + 画出候选的全部框。

为什么需要它：原 `_viz_qwen3vl.py` 只标问句与 IoU、**没有图例**，人工判读时
容易把红/绿归错模型；且它只画最终选中的那一个框，看不到模型其实返回了多个。

口径（与原图一致）：绿 = 当前锁定提交（基线）；红 = Qwen3-VL。
Qwen 的候选框全部画出并编号 Q1/Q2…，其中 Q1 是流水线实际采用的那个。

用法：
    python _viz_recheck.py 000002_002 000002_005
输出 viz_out/recheck/{qid}.png
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402

OUT = R / "viz_out" / "recheck"
PROBE = R / "qwen3vl_probe.json"
RE_BBOX = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
FONTS = [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf",
         r"C:\Windows\Fonts\consola.ttf"]
GREEN, RED = (0, 255, 0), (255, 0, 0)


def font(size: int):
    """加载可用字体，失败回退 PIL 默认字体。"""
    for p in FONTS:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def boxes_from_raw(raw: str) -> list:
    """从模型 raw 原文里取出全部候选框（探针的 raw 是完整的，未被截断）。"""
    return [[int(a) / 1000, int(b) / 1000, int(c) / 1000, int(d) / 1000]
            for a, b, c, d in RE_BBOX.findall(raw or "")]


def chip(d: ImageDraw.ImageDraw, xy, text: str, fill, f) -> None:
    """在指定位置画一个带底色的小标签。"""
    x, y = xy
    w = d.textlength(text, font=f)
    d.rectangle([x, y, x + w + 8, y + f.size + 6], fill=fill)
    d.text((x + 4, y + 3), text, fill=(255, 255, 255) if fill != GREEN
           else (0, 0, 0), font=f)


def render(qid: str, base: dict, probe: dict) -> bool:
    """渲染单条复核图。"""
    rel = base[qid]["visible"]
    p = F.resolve_img(rel)
    if p is None:
        print(f"  [skip] 图不存在 {qid} {rel}")
        return False
    im = Image.open(p).convert("RGB")
    W, H = im.size
    bb = base[qid]["bbox"]
    rec = probe.get(qid) or {}
    qb_all = boxes_from_raw(rec.get("raw", ""))
    if not qb_all and rec.get("bbox"):
        qb_all = [rec["bbox"]]

    pad = 96
    canvas = Image.new("RGB", (W, H + pad + 30), (255, 255, 255))
    canvas.paste(im, (0, pad))
    d = ImageDraw.Draw(canvas)
    f_t, f_c = font(24), font(20)
    iou = F.iou(bb, qb_all[0]) if qb_all else None
    d.text((8, 6), f"{qid}   {base[qid]['query']}", fill=(0, 0, 0), font=f_t)
    d.text((8, 36), f"绿(细)=基线 面积{F.area(bb):.4f}   "
                    f"红=Qwen3-VL 共{len(qb_all)}个候选   "
                    f"IoU(绿, Q1)={'—' if iou is None else f'{iou:.3f}'}",
           fill=(120, 0, 0), font=f_c)
    d.text((8, 62), "Q1 = 流水线实际采用（取首个框）；实线=Q1，虚线语义见标签",
           fill=(90, 90, 90), font=f_c)

    lw = max(3, int(min(W, H) * 0.005))
    # 基线（绿）
    d.rectangle([bb[0] * W, bb[1] * H + pad, bb[2] * W, bb[3] * H + pad],
                outline=GREEN, width=lw)
    chip(d, (bb[0] * W, max(pad, bb[1] * H + pad - f_c.size - 6)), "绿 基线",
         GREEN, f_c)
    # Qwen 全部候选（红）
    for i, q in enumerate(qb_all, 1):
        d.rectangle([q[0] * W, q[1] * H + pad, q[2] * W, q[3] * H + pad],
                    outline=RED, width=lw + (2 if i == 1 else 0))
        chip(d, (q[0] * W, min(H + pad - f_c.size - 8,
                               q[1] * H + pad + 2)), f"Q{i}", RED, f_c)
    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT / f"{qid}.png")
    print(f"  {qid}  {len(qb_all)} 个 Qwen 候选  → {OUT / (qid + '.png')}")
    return True


def main() -> int:
    qids = sys.argv[1:] or ["000002_002", "000002_005"]
    base = json.loads(F.BASE_JSON.read_text(encoding="utf-8"))
    probe = json.loads(PROBE.read_text(encoding="utf-8")) \
        if PROBE.exists() else {}
    for q in qids:
        if q not in base:
            print(f"  [skip] 键不存在 {q}")
            continue
        render(q, base, probe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
