# -*- coding: utf-8 -*-
"""热成像类差异条目的对照图：左=可见光图(画 v1 的框)，右=红外图(画 v2/v3 的框)。

用途：v1 与 v2/v3 的差异全在这批条目上，000022 那种四版一致的图看不出区别。
所有框一律按**真实坐标**绘制——不引入任何偏移，避免造出不存在的优劣。
"""
import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))
import _sub_path as SP  # noqa: E402
D = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
SUB = R / "submission_zips" / "semifinal"  # noqa: F401  路径统一走 _sub_path
OUT = R / "viz_thermal_diff"
TOPN = 6
PW, PH = 960, 540          # 单幅面板尺寸（原图 1920x1080 的一半）
BAND = 132                 # 顶部说明带高度

C_V1 = (255, 40, 40)       # v1（可见光框）红
C_IR = (0, 220, 0)         # v2/v3（红外框）绿


def font(size):
    for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", default="extreme",
                    choices=["extreme", "typical", "random"],
                    help="extreme=IoU 最小；typical=IoU 居中的代表样本；random=随机")
    ap.add_argument("--n", type=int, default=TOPN)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outdir = Path(a.out) if a.out else OUT
    outdir.mkdir(parents=True, exist_ok=True)

    q = json.loads((D / "queries" / "queries.json").read_text(encoding="utf-8"))
    v1 = json.loads(SP.find("result_final_semi").read_text(encoding="utf-8"))
    v2 = json.loads(SP.find("result_v2_swap33").read_text(encoding="utf-8"))
    ir = json.loads((R / "_semi_retry_thermal.json").read_text(encoding="utf-8"))

    diffs = []
    for k, rec in ir.items():
        b_ir = rec.get("bbox")
        b_v1 = v1.get(k, {}).get("bbox")
        if b_ir and b_v1:
            g = iou(b_v1, b_ir)
            if g < 0.5:
                diffs.append((g, k))
    diffs.sort()
    if a.pick == "extreme":
        sel = diffs[:a.n]
    elif a.pick == "typical":
        # 取 IoU 分布中段（剔除两端各 1/4），再均匀取样，代表"普通差异"而非极端
        lo = len(diffs) // 4
        hi = len(diffs) - len(diffs) // 4
        mid = diffs[lo:hi]
        step = max(1, len(mid) // a.n)
        sel = mid[::step][:a.n]
    else:
        random.seed(a.seed)
        sel = random.sample(diffs, min(a.n, len(diffs)))
    print(f"实质差异 {len(diffs)} 条 | 选取方式={a.pick} | 出图 {len(sel)} 张")
    print(f"IoU 范围: {min(g for g,_ in sel):.3f} ~ {max(g for g,_ in sel):.3f}\n")

    f_t = font(26)
    f_l = font(21)
    for g, k in sel:
        b1 = v1[k]["bbox"]
        b2 = v2[k]["bbox"]
        canvas = Image.new("RGB", (PW * 2, BAND + PH), (18, 18, 18))
        d = ImageDraw.Draw(canvas)

        panels = (("visible", b1, C_V1, "可见光图 + v1 框"),
                  ("infrared", b2, C_IR, "红外图 + v2/v3 框(IR)"))
        for pi, (mod, box, col, capt) in enumerate(panels):
            im = Image.open(D / q[k][mod]).convert("RGB").resize((PW, PH))
            canvas.paste(im, (pi * PW, BAND))
            dd = ImageDraw.Draw(canvas)
            if box:
                x1, y1 = box[0]*PW + pi*PW, box[1]*PH + BAND
                x2, y2 = box[2]*PW + pi*PW, box[3]*PH + BAND
                dd.rectangle([x1, y1, x2, y2], outline=col, width=4)
            dd.rectangle([pi*PW + 6, BAND + 6,
                          pi*PW + 6 + int(dd.textlength(capt, font=f_l)) + 12,
                          BAND + 6 + f_l.size + 10], fill=(0, 0, 0))
            dd.text((pi*PW + 12, BAND + 11), capt, fill=col, font=f_l)
            if pi == 1:
                dd.line([(PW, BAND), (PW, BAND + PH)], fill=(90, 90, 90), width=3)

        lines = [f"{k}   IoU(v1,IR)={g:.3f}",
                 f"问句: {q[k]['query'][:96]}",
                 f"v1  (可见光) : {[round(x, 4) for x in b1]}",
                 f"v2/v3 (红外) : {[round(x, 4) for x in b2]}"]
        y = 8
        for i, s in enumerate(lines):
            d.text((12, y), s, fill=(255, 255, 255) if i < 2 else
                   (C_V1 if i == 2 else C_IR), font=(f_t if i == 0 else f_l))
            y += (f_t if i == 0 else f_l).size + 6
        p = outdir / f"{k}_diff.png"
        canvas.save(p)
        print(f"  已写 {p.name}  IoU={g:.3f}")

    print(f"\n目录 {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
