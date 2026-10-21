# -*- coding: utf-8 -*-
"""把各 zip 对 000022.png 各问题的框画在同一张图上，便于直观对比。

每个问题出一张图，四份提交件用不同颜色叠加，并附图例（含数值框）。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))
import _sub_path as SP  # noqa: E402
SUB = R / "submission_zips" / "semifinal"  # noqa: F401  路径统一走 _sub_path
OUT = R / "viz_000022"
IMG_REL = "Images/visible/000022.png"

# 名字 -> 提交件 JSON（按对比需要排列）
ZIPS = {
    "qwen8b_baseline": "result_qwen8b_semi",
    "v1_final": "result_final_semi",
    "v2_swap33": "result_v2_swap33",
    "v3_swap58": "result_v3_swap58",
}
COLORS = {
    "qwen8b_baseline": (255, 140, 0),     # 橙
    "v1_final": (255, 0, 0),              # 红
    "v2_swap33": (0, 200, 0),             # 绿
    "v3_swap58": (0, 120, 255),           # 蓝
}


def font(size: int):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = json.loads(
        (R / "复赛数据集-基于大模型的多模态视觉理解与推理" / "queries"
         / "queries.json").read_text(encoding="utf-8"))
    data = {n: json.loads(SP.find(stem).read_text(encoding="utf-8"))
            for n, stem in ZIPS.items()}
    src = R / "复赛数据集-基于大模型的多模态视觉理解与推理" / IMG_REL
    base = Image.open(src).convert("RGB")
    W, H = base.size
    print(f"底图 {IMG_REL} {W}x{H}")

    qids = [k for k, v in queries.items() if v["visible"] == IMG_REL]
    print(f"该图问题数: {len(qids)}\n")

    for qid in qids:
        im = base.copy()
        d = ImageDraw.Draw(im)
        f_t = font(int(H * 0.030))
        f_l = font(int(H * 0.024))
        lines = [f"{qid}   {queries[qid]['query']}", ""]
        vals = {}
        for name in ZIPS:
            b = data[name][qid]["bbox"]
            vals[name] = tuple(b) if b else None
            lines.append(f"  {name}: {[round(x, 4) for x in b] if b else '(无框)'}")
        n_uniq = len(set(vals.values()))
        lines.insert(1, f"  ※ 本问题共 {n_uniq} 种不同答案"
                        + ("（四版完全相同 → 只画一个框）" if n_uniq == 1 else ""))

        # 只按**真实坐标**绘制：数值相同的版本合并为一个框，绝不加偏移。
        # 教训：早期版本给不同变体加逐层外扩以"显示重合"，结果让零偏移的那版
        # 看起来最紧最准，凭空造出优劣排序——可视化绝不能引入系统性偏差。
        groups = {}
        for name in ZIPS:
            b = vals[name]
            groups.setdefault(b, []).append(name)
        for gi, (b, names) in enumerate(groups.items()):
            if not b:
                continue
            x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
            col = COLORS[names[0]]
            d.rectangle([x1, y1, x2, y2], outline=col, width=5)
            tag = "+".join(n.split("_")[0] for n in names)
            d.text((x1 + 4, max(0, y1 - f_l.size - 6)), f"{tag} {gi+1}",
                   fill=col, font=f_l) if len(groups) > 1 else \
                d.text((x1 + 4, max(0, y1 - f_l.size - 6)),
                       f"{tag}（{len(names)} 版相同）", fill=col, font=f_l)
        # 图例配色随实际分组
        for i in range(3, len(lines)):
            nm = list(ZIPS)[i - 3]
            lines[i] = f"  {nm}: {[round(x, 4) for x in vals[nm]] if vals[nm] else '(无框)'}"

        pad = 12
        line_h = [f_t.size + 8] + [f_l.size + 8] * (len(lines) - 1)
        tw = max(d.textlength(s, font=(f_t if i == 0 else f_l))
                 for i, s in enumerate(lines)) + pad * 2
        th = sum(line_h) + pad
        d.rectangle([0, 0, tw, th], fill=(0, 0, 0))
        y = pad // 2
        for i, s in enumerate(lines):
            ft = f_t if i == 0 else f_l
            if i == 1:
                col = (255, 255, 0)
            elif i <= 2:
                col = (255, 255, 255)
            else:
                col = COLORS[list(ZIPS)[i - 3]]
            d.text((pad, y), s, fill=col, font=ft)
            y += ft.size + 8
        p = OUT / f"{qid}_000022.png"
        im.save(p)
        print(f"  已写 {p.name}（{n_uniq} 种答案）")

    # 另出一张"全问题叠加"总览：每题一色
    im = base.copy()
    d = ImageDraw.Draw(im)
    pal = [(255, 0, 0), (0, 200, 0), (0, 120, 255), (255, 140, 0), (200, 0, 200)]
    f_l = font(int(H * 0.026))
    lines = [f"{IMG_REL} — v2_swap33 全部 {len(qids)} 个问题", ""]
    for i, qid in enumerate(qids):
        b = data["v2_swap33"][qid]["bbox"]
        if not b:
            continue
        col = pal[i % len(pal)]
        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        d.rectangle([x1, y1, x2, y2], outline=col, width=6)
        d.text((x1 + 6, max(0, y1 - f_l.size - 6)), qid, fill=col, font=f_l)
        lines.append(f"  {qid} {queries[qid]['query'][:58]}")
    pad = 12
    tw = max(d.textlength(s, font=f_l) for s in lines) + pad * 2
    th = sum(f_l.size + 8 for _ in lines) + pad
    d.rectangle([0, 0, tw, th], fill=(0, 0, 0))
    y = pad // 2
    for i, s in enumerate(lines):
        d.text((pad, y), s, fill=(255, 255, 255) if i <= 1 else pal[(i - 2) % len(pal)],
               font=f_l)
        y += f_l.size + 8
    p = OUT / "_overview_000022_v2_swap33.png"
    im.save(p)
    print(f"\n已写总览 {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
