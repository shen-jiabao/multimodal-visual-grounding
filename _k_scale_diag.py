#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""K 件输出尺度的判定（本地、零 GPU）。

问题
    Qwen2.5-VL 的原生接地约定与提示词要求（"整数、归一化到 0-1000"）不一致：
    冒烟输出里出现 >1.0 的值（如 2.000 / 1.091）。G2 的"可解析"判据抓不到——
    **越界框照样能解析**，而"不越界但整体错位"更坏（会被当合法改动接受）。

做法
    ★ 从闸门产物里的 `raw` 字段（模型原始输出文本）**重新解析整数**，
      而不是用它已解析好的 `bbox` —— 后者在适配器生效后已经是归一化值，
      再拿去做"÷1000 vs ÷(W,H)"的比较会得出错误结论（本脚本曾因此误报）。
    三个假设各算一次与交付件框的 IoU，哪个系统性对得上即为真尺度：

        H1 = 0-1000：      box = v / 1000
        H2 = 原图像素：     box = v / (W, H)
        H3 = smart-resize：box = v / (W', H')   （Qwen2.5-VL 边长取整到 28 的倍数）

用法
    python _k_scale_diag.py                      # 缺省读 <ROOT>/_k_gate20.json
    python _k_scale_diag.py --gate <另一份产物>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sub_path as SP  # noqa: E402

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
# 20 条冒烟的产物（含模型原始输出文本）——本判据的证据，随包提交
GATE = R / "_k_gate20.json"


def iou(a, b) -> float:
    """两个归一化框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    s = ar + br - inter
    return inter / s if s > 0 else 0.0


def clip(b) -> list[float]:
    """裁剪到 [0,1] 并保证有序。"""
    x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in b)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def smart_resize_wh(w: int, h: int, factor: int = 28,
                    min_px: int = 4 * 28 * 28,
                    max_px: int = 16384 * 28 * 28) -> tuple[int, int]:
    """复刻 Qwen2.5-VL 的 smart_resize 尺寸计算（只算目标边长，不真的缩放）。

    规则（与 transformers 的 Qwen2VLImageProcessor 同）：
        先把 h/w 取整到 factor 的倍数；若总像素超过 max_px（或低于 min_px），
        按面积比例整体缩放后再取整到 factor 的倍数。
    """
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_px:
        beta = (h * w / max_px) ** 0.5
        hb = max(factor, int(h / beta / factor) * factor)
        wb = max(factor, int(w / beta / factor) * factor)
    elif hb * wb < min_px:
        beta = (min_px / (h * w)) ** 0.5
        hb = max(factor, int(h * beta / factor) * factor)
        wb = max(factor, int(w * beta / factor) * factor)
    return wb, hb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default=str(GATE),
                    help="20 条冒烟产物（须含模型原始输出 `raw`）；"
                         "缺省 <ROOT>/_k_gate20.json")
    a = ap.parse_args()

    import _qwen_full_pipeline as Q   # 复用同一个框正则（判据只实现一处）
    q = json.loads((DATA / "queries" / "queries.json").read_text("utf-8"))
    REF = json.loads(SP.find("result_semi_q38_nonord").read_text("utf-8"))
    gate = json.loads(Path(a.gate).read_text("utf-8"))

    print(f"参照件 {SP.find('result_semi_q38_nonord').name}｜冒烟 {len(gate)} 条\n")
    print(f"{'qid':<13}{'原图WxH':>12}{'sr WxH':>12}  "
          f"{'H1':>7}{'H2':>7}{'H3':>7}  {'参照框':<30}{'模型原始整数':<26}")
    rows = []
    for k, v in gate.items():
        ms = list(Q.RE_BBOX.finditer(v.get("raw") or ""))
        if not ms:
            print(f"{k:<13}{'—':>12}{'—':>12}  {'无框':>21}")
            continue
        raw = [int(g) for g in ms[0].groups()]
        img = DATA / q[k]["visible"]
        with Image.open(img) as im:
            w, h = im.size
        sw, sh = smart_resize_wh(w, h)
        h1 = clip([raw[0] / 1000, raw[1] / 1000, raw[2] / 1000, raw[3] / 1000])
        h2 = clip([raw[0] / w, raw[1] / h, raw[2] / w, raw[3] / h])
        h3 = clip([raw[0] / sw, raw[1] / sh, raw[2] / sw, raw[3] / sh])
        ref = REF[k]["bbox"]
        i1, i2, i3 = iou(h1, ref), iou(h2, ref), iou(h3, ref)
        rows.append((k, i1, i2, i3))
        print(f"{k:<13}{f'{w}x{h}':>12}{f'{sw}x{sh}':>12}  "
              f"{i1:>7.3f}{i2:>7.3f}{i3:>7.3f}  "
              f"{str([round(x, 3) for x in ref]):<30}{str(raw):<26}")

    if not rows:
        print("无可用行")
        return 1
    n = len(rows)
    med = [sorted(r[i] for r in rows)[n // 2] for i in (1, 2, 3)]
    print(f"\n中位 IoU（与参照框比）：H1={med[0]:.3f}  H2={med[1]:.3f}  "
          f"H3={med[2]:.3f}")
    print(f"逐条更优计数：H2>H1 {sum(1 for r in rows if r[2] > r[1])}/{n}｜"
          f"H3>H2 {sum(1 for r in rows if r[3] > r[2])}/{n}｜"
          f"H3≥0.9 的条数 {sum(1 for r in rows if r[3] >= 0.9)}/{n}")

    print("\n判读：")
    if med[2] > med[1] and med[2] > 0.6:
        print("  ⇒ **H3 成立：坐标空间 = smart-resize 尺寸（28 的倍数）**")
        print("    修法：解析后按 smart_resize 的 (W',H') 归一化再裁剪到 [0,1]，")
        print("          并经 `QWEN_BBOX_SPACE=smart_resize_px` 注入管线。")
    elif med[1] > med[0] and med[1] > 0.3:
        print("  ⇒ H2 成立（原图像素）；H3 未胜出。")
    else:
        print("  ⇒ 三个假设都不成立，需进一步诊断。")
    print(f"  ⚠ 无论哪种，都不能直接用 ÷1000：H1 中位 {med[0]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
