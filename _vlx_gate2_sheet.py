# -*- coding: utf-8 -*-
"""把「方案乙」要判的 8 张拼成一张联系表（一图看完，不用逐个开文件）。

每格内容
    标题行：qid + 类别 + IoU(与基线) + 面积比
    主图　：红粗框 = VLX 选中；绿细框 = 基线（0.7766 件）
    右下角：红框邻域 3 倍放大（部分条目框极小，缩图后看不清）

为什么重新渲染而不是直接用 _vlx_vis20/*.jpg：
    那些图里红框是原图 1920 宽的 4px，缩到 900 宽只剩 2px，判不准。
    这里按坐标重画，并把基线框一并画上（人眼可同时读"分歧在哪"）。

用法
    python -B _vlx_gate2_sheet.py          # 8 格：4 条分歧 + 4 条抽验
    python -B _vlx_gate2_sheet.py --all    # 20 格：闸门② 精确计数用

★ 为什么要有 --all：
    仅抽验 4 条无法支撑"那 14 条里错几条"的结论——n=4 且全对时，
    失败率的 95% 置信上界仍高达 52.7%，等于什么都没说。
    闸门阈值是 ≥12/20，而分歧 6 条里已知错 2~3 条 ⇒ 结论取决于那 14 条，
    要**精确**计数只能全看。20 条拼一张，看一遍约 20 秒。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"

# (qid, 角色)
PANELS = [
    ("000001_004", "分歧-非序"),
    ("000001_006", "分歧-非序(复数)"),
    ("000246_006", "分歧-序数(域外)"),
    ("000113_004", "分歧-序数(域外)"),
    ("000001_005", "抽验-与基线一致"),
    ("000015_001", "抽验-与基线一致"),
    ("000015_002", "抽验-与基线一致"),
    ("000015_004", "抽验-与基线一致"),
]


def get_font(size: int):
    """取一个可用的 TrueType 字体；失败退回默认位图字体。"""
    for name in ("consola.ttf", "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap(text: str, n: int) -> list[str]:
    """按词折行。"""
    out, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) + 1 <= n:
            cur = (cur + " " + w).strip()
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out[:2]


def build(panels, pw: int, zoom: int, cap: int, cols: int,
          out: Path, res: dict, base: dict, cores: set) -> None:
    """渲染联系表：红粗框=VLX、绿细框=基线、右下角放大。"""
    f_cap, f_q = get_font(max(14, cap // 3)), get_font(max(13, cap // 3 - 2))
    tiles = []
    for qid, role in panels:
        v = res[qid]
        b, bb = v.get("bbox"), base[qid]["bbox"]
        img = Image.open(SEMI / "Images" / "visible" / v["img"]).convert("RGB")
        W, H = img.size
        d = ImageDraw.Draw(img)

        def px(box):
            return [box[0] * W, box[1] * H, box[2] * W, box[3] * H]

        gb = px(bb)
        d.rectangle(gb, outline=(0, 220, 0), width=4)          # 绿 = 基线
        crop = None
        if b:
            rb = px(b)
            d.rectangle(rb, outline=(255, 0, 0), width=6)      # 红 = VLX
            u = [min(rb[0], gb[0]), min(rb[1], gb[1]),
                 max(rb[2], gb[2]), max(rb[3], gb[3])]
            pad = 0.15 * max(u[2] - u[0], u[3] - u[1], 60)
            u = [max(0, u[0] - pad), max(0, u[1] - pad),
                 min(W, u[2] + pad), min(H, u[3] + pad)]
            crop = img.crop(tuple(int(x) for x in u))
            crop.thumbnail((zoom, zoom), Image.LANCZOS)

        main_img = img.resize((pw, int(pw * H / W)), Image.LANCZOS)
        tile = Image.new("RGB", (pw, cap + main_img.height), (255, 255, 255))
        tile.paste(main_img, (0, cap))
        td = ImageDraw.Draw(tile)
        if crop is not None:
            zx = pw - crop.width - 10
            zy = cap + main_img.height - crop.height - 10
            tile.paste(crop, (zx, zy))
            td.rectangle([zx, zy, zx + crop.width, zy + crop.height],
                         outline=(255, 0, 0), width=2)

        kind = "序数" if qid in cores else "非序数"
        io = F.iou(b, bb) if b else 0.0
        ratio = (F.area(b) / F.area(bb)) if b and F.area(bb) else 0.0
        tag = "一致" if io >= 0.5 else "分歧"
        td.text((5, 2), f"{qid}  [{kind}]  {role}", (0, 0, 0), font=f_cap)
        td.text((5, cap // 2 + 2), f"IoU {io:.3f} 面积比 {ratio:.2f}x [{tag}]",
                (150, 0, 0), font=f_q)
        for i, ln in enumerate(wrap(v["query"], int(pw / 8.6))):
            td.text((pw - int(pw * 0.52), 2 + i * (cap // 2 + 2)),
                    ln, (0, 0, 160), font=f_q)
        tiles.append(tile)

    rows = (len(tiles) + cols - 1) // cols
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * pw + (cols + 1) * 8,
                              rows * th + (rows + 1) * 8), (232, 232, 232))
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (8 + c * (pw + 8), 8 + r * (th + 8)))
    sheet.save(out, quality=88)
    print(f"已写 {out}")
    print(f"  {sheet.width} x {sheet.height}，{len(tiles)} 格，"
          f"{out.stat().st_size / 1e6:.1f} MB")
    print("  红粗框 = VLX 选中；绿细框 = 基线；右下角小图 = 红绿并集放大")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="出全 20 格（闸门② 精确计数用）")
    a = ap.parse_args()

    res = json.loads((R / "_vlx_smoke20_out.json").read_text(encoding="utf-8"))
    base = json.loads(SP.find("result_semi_n32_orddirect").read_text(
        encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(
        encoding="utf-8")))

    if a.all:
        panels = [(k, "全量核验") for k in res]
        build(panels, 700, 250, 56, 4, R / "_vlx_gate2_all20.jpg",
              res, base, cores)
    else:
        build(PANELS, 900, 320, 62, 2, R / "_vlx_gate2_sheet.jpg",
              res, base, cores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

