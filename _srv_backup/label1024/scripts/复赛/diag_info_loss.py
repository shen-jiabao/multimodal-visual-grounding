# -*- coding: utf-8 -*-
"""输入侧信息损失审计（只读）。

回答一个问题：有没有信息在**送进模型之前**就丢了，且丢了就补不回来。

统计三类可量化的损失，全部只读图像，不加载模型、不改任何预测：

  1. 分辨率 / 视觉 token —— 图像是否被 processor 降采样；
  2. 深度动态范围     —— 固定 0~19999mm 归一化后，场景内容只占几个灰度级；
  3. 深度语义冲突     —— 无效像素渲染为 0（纯黑），与「很近」的灰度不可区分。

用法::

    python scripts/复赛/diag_info_loss.py
    python scripts/复赛/diag_info_loss.py --depth-sample 400 --json outputs/复赛/diag/info_loss.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import inspect
import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEPTH_MAX = 19999  # 与 src/data/preprocess.py 的 DEPTH_VALID_MAX 保持一致


def smart_resize(h: int, w: int, factor: int = 32,
                 min_pixels: int = 128 * 32 * 32,
                 max_pixels: int = 32 * 32 * 768) -> tuple[int, int]:
    """复刻 HF 的 smart_resize：先按像素预算缩放，再对齐到 factor 的整数倍。

    factor = patch_size(16) × merge_size(2) = 32，即每 32×32 像素产 1 个视觉 token。
    """
    hb = max(factor, int(round(h / factor) * factor))
    wb = max(factor, int(round(w / factor) * factor))
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, int(math.floor(h / beta / factor)) * factor)
        wb = max(factor, int(math.floor(w / beta / factor)) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = max(factor, int(math.ceil(h * beta / factor)) * factor)
        wb = max(factor, int(math.ceil(w * beta / factor)) * factor)
    return hb, wb


def tokens(h: int, w: int, max_pixels: int) -> tuple[int, int, int]:
    hb, wb = smart_resize(h, w, max_pixels=max_pixels)
    return hb, wb, (hb // 32) * (wb // 32)


# ---------------------------------------------------------------- 1. 分辨率
def audit_resolution(root: str) -> dict:
    out = {}
    for name in ("visible", "infrared", "depth"):
        files = sorted(glob.glob(os.path.join(root, "Images", name, "*")))
        cnt = collections.Counter()
        modes = collections.Counter()
        for f in files:
            try:
                with Image.open(f) as im:
                    cnt[im.size] += 1          # (W, H)，仅读文件头
                    modes[im.mode] += 1
            except Exception:  # noqa: BLE001
                cnt[("ERR", 0)] += 1
        out[name] = {
            "files": len(files),
            "sizes": {f"{w}x{h}": c for (w, h), c in cnt.most_common()},
            "modes": dict(modes),
        }
    return out


# ---------------------------------------------------------------- 2/3. 深度
def audit_depth(root: str, sample: int, seed: int = 0) -> dict:
    pngs = sorted(glob.glob(os.path.join(root, "Images", "depth", "*.png")))
    jpgs = sorted(glob.glob(os.path.join(root, "Images", "depth", "*.jpg")))
    rng = random.Random(seed)
    pick = rng.sample(pngs, min(sample, len(pngs)))

    inv, span, near_black, used_levels = [], [], [], []
    worst = []
    for f in pick:
        a = np.asarray(Image.open(f))
        if a.ndim != 2:                      # 兼容异常通道数，先降为灰度
            a = a[..., 0]
        a = a.astype(np.float32)
        valid = a > 0
        if valid.sum() < 100:
            continue
        v = a[valid]
        inv.append(1.0 - valid.mean())
        p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
        span.append((p99 - p1) / DEPTH_MAX * 255.0)
        # 渲染为 <8 灰度的有效像素：与「无效=0=纯黑」在视觉上不可分
        near_black.append(float((v < 8.0 / 255.0 * DEPTH_MAX).mean()))
        used_levels.append(float((p99 - p1) / DEPTH_MAX * 255.0))
        worst.append((float(span[-1]), os.path.basename(f)))

    inv = np.array(inv)
    span = np.array(span)
    nb = np.array(near_black)
    worst.sort()
    return {
        "png_files": len(pngs),
        "jpg_files": len(jpgs),
        "sampled": int(inv.size),
        "invalid_ratio": {
            "mean": float(inv.mean()), "median": float(np.median(inv)),
            "gt_50pct": int((inv > 0.5).sum()),
            "gt_90pct": int((inv > 0.9).sum()),
        },
        "span_gray_current": {
            "median": float(np.median(span)), "p10": float(np.percentile(span, 10)),
            "min": float(span.min()),
        },
        "span_lt_32": int((span < 32).sum()),
        "span_lt_64": int((span < 64).sum()),
        "span_lt_128": int((span < 128).sum()),
        "stretch_gain_median": float(255.0 / max(np.median(span), 1e-6)),
        "valid_px_gray_lt_8": {
            "mean": float(nb.mean()), "median": float(np.median(nb)),
            "gt_10pct": int((nb > 0.10).sum()),
        },
        "worst_scenes": [{"span": round(s, 1), "file": n} for s, n in worst[:5]],
    }


# ---------------------------------------------------------------- 4. 红外
def audit_infrared(root: str, sample: int, seed: int = 0) -> dict:
    files = sorted(glob.glob(os.path.join(root, "Images", "infrared", "*")))
    rng = random.Random(seed)
    pick = rng.sample(files, min(sample, len(files)))
    gray_ok, levels, ent = 0, [], []
    for f in pick:
        a = np.asarray(Image.open(f))
        if a.ndim == 3 and a.shape[2] >= 3:
            if np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2]):
                gray_ok += 1
        g = a[..., 0] if a.ndim == 3 else a
        hist = np.bincount(g.ravel().astype(np.uint16), minlength=256).astype(np.float64)
        p = hist / max(hist.sum(), 1)
        nz = p[p > 0]
        levels.append(float((hist > 0).sum()))
        ent.append(float(-(nz * np.log2(nz)).sum()))
    return {
        "files": len(files),
        "sampled": len(pick),
        "r_eq_g_eq_b": gray_ok,
        "levels_used_median": float(np.median(levels)) if levels else 0.0,
        "entropy_bits_median": round(float(np.median(ent)), 2) if ent else 0.0,
    }


# ---------------------------------------------------------------- 报告
def main() -> None:
    ap = argparse.ArgumentParser(description="输入侧信息损失审计（只读）")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--depth-sample", type=int, default=300)
    ap.add_argument("--ir-sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    res = audit_resolution(args.root)
    print("=" * 74)
    print("1. 分辨率分布（视觉 token 预算的上限）")
    for name, d in res.items():
        print(f"  {name:9s} {d['files']:5d} 个   modes={d['modes']}")
        for k, c in list(d["sizes"].items())[:4]:
            print(f"      {k:>12s}  ×{c}")
    print("\n  每 32×32 像素 = 1 个视觉 token（patch16 × merge2），三图合计：")
    print(f"  {'max_pixels 假设':<34s} {'1920x1080':>22s} {'640x360':>16s}")
    for label, mp in (("代码默认视频 32*32*768=786432", 32 * 32 * 768),
                      ("Qwen2.5 图像 28*28*1280=1003520", 28 * 28 * 1280),
                      ("图像上限 28*28*2*30000=47.04M", 28 * 28 * 2 * 30000),
                      ("不限", 10 ** 12)):
        _, _, t1 = tokens(1080, 1920, mp)
        _, _, t2 = tokens(360, 640, mp)
        print(f"  {label:<34s} {t1:>6d}/图 ×3={3 * t1:>6d} {t2:>6d}/图 ×3={3 * t2:>5d}")

    dep = audit_depth(args.root, args.depth_sample, args.seed)
    print("\n" + "=" * 74)
    print(f"2/3. 深度渲染（随机 {dep['sampled']} 张 PNG / 共 {dep['png_files']} 张）")
    i = dep["invalid_ratio"]
    print(f"  无效像素占比   mean {i['mean']:.3f}   median {i['median']:.3f}   "
          f">50% 的 {i['gt_50pct']} 张   >90% 的 {i['gt_90pct']} 张")
    s = dep["span_gray_current"]
    print(f"  场景内容占用的灰度级数（固定 0~19999mm 归一化）")
    print(f"      median {s['median']:.1f}   p10 {s['p10']:.1f}   min {s['min']:.1f}")
    print(f"      <32 级 {dep['span_lt_32']} 张   <64 级 {dep['span_lt_64']} 张   "
          f"<128 级 {dep['span_lt_128']} 张")
    print(f"      按 p1..p99 拉伸到 0~255 后，中位可提升 {dep['stretch_gain_median']:.1f}×")
    nb = dep["valid_px_gray_lt_8"]
    print(f"  有效像素中渲染灰度 <8（与「无效=黑」不可区分）")
    print(f"      mean {nb['mean']:.3f}   median {nb['median']:.3f}   "
          f">10% 的场景 {nb['gt_10pct']} 张")
    print("  最暗的 5 个场景：" + ", ".join(
        f"{w['file']}({w['span']}级)" for w in dep["worst_scenes"]))

    ir = audit_infrared(args.root, args.ir_sample, args.seed)
    print("\n" + "=" * 74)
    print(f"4. 红外（随机 {ir['sampled']} 张 / 共 {ir['files']} 张）")
    print(f"  R==G==B 的 {ir['r_eq_g_eq_b']}/{ir['sampled']}（真灰度堆叠，无色信息）")
    print(f"  实际用到的灰度级数 median {ir['levels_used_median']:.0f}/256   "
          f"信息熵 median {ir['entropy_bits_median']} bit")

    print("\n" + "=" * 74)
    print("5. 读取/清洗环节（源码速览）")
    for mod, fn in (("src.utils.io", "read_image_uint16"),
                    ("src.utils.text", "clean_query")):
        try:
            m = __import__(mod, fromlist=[fn])
            src = inspect.getsource(getattr(m, fn))
            print(f"\n  --- {mod}.{fn} ---")
            print("\n".join("  " + l for l in src.splitlines()))
        except Exception as e:  # noqa: BLE001
            print(f"  [{mod}.{fn}] 不可读：{e}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"resolution": res, "depth": dep, "infrared": ir},
                      fh, ensure_ascii=False, indent=1)
        print(f"\n已导出：{args.json}")


if __name__ == "__main__":
    main()
