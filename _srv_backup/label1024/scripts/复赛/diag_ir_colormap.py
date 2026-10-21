# -*- coding: utf-8 -*-
"""红外「伪彩 → 灰度」信息损失审计（只读）。

背景：`src/data/fusion.py::infrared_pseudo_rgb` 在 split 模式下把红外图送进模型前
会先转灰度（`denoise_infrared` 对三通道取 `np.mean`）。这一步隐含一个假设：

    红外三通道「视觉特征高度一致」→ 转灰度不丢信息。

本脚本检验该假设。若红外是**伪彩渲染**（温度 → 颜色映射，如 jet / inferno），
则不同温度可能映到**同一灰度**（例：纯蓝 (0,0,255)、纯绿 (0,255,0)、纯红 (255,0,0)
的平均灰度都是 85），温度序被压平 —— 这是**真的**信息损失。

判据（每张图独立计算，不依赖温度真值）：

  1. `pc1_var`   : RGB 三通道的 PC1 方差占比。真伪彩图 ≈ 1.0（像素落在一条 1-D 曲线上）；
  2. `rho_gray`  : 灰度与 PC1（温度代理）的 |Spearman|。≈1 表示灰度保序，≈0 表示序被破坏；
  3. `spread`    : 同一灰度档内 PC1 的相对跨度。≈0 表示每个灰度对应唯一温度（无损）。

用法::

    python scripts/复赛/diag_ir_colormap.py
    python scripts/复赛/diag_ir_colormap.py --sample 200 --json outputs/复赛/diag/ir_colormap.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MAX_PX = 40_000          # 每图抽样像素上限（SVD 成本 O(n)）


def _rank(x: np.ndarray) -> np.ndarray:
    """平均秩（忽略并列细节，作为 Spearman 的代理足够）。"""
    order = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    r[order] = np.arange(x.size, dtype=np.float64)
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def analyze_one(path: str, rng: random.Random) -> dict | None:
    a = np.asarray(Image.open(path))
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    a = a[..., :3]
    flat = a.reshape(-1, 3)
    if flat.shape[0] > MAX_PX:
        sel = np.array(rng.sample(range(flat.shape[0]), MAX_PX))
        flat = flat[sel]
    if flat.shape[0] < 100:
        return None

    rgb = flat.astype(np.float64)
    gray = flat.mean(axis=1)                                   # 现行做法
    lum = (0.299 * flat[:, 0] + 0.587 * flat[:, 1] + 0.114 * flat[:, 2])
    r_ch = flat[:, 0].astype(np.float64)

    x = rgb - rgb.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    pc1 = x @ vt[0]
    pc1_var = float(var[0] / var.sum()) if var.sum() > 0 else 0.0

    # 归一化到 [0,1] 以便跨图比较
    lo, hi = float(pc1.min()), float(pc1.max())
    pc1n = (pc1 - lo) / (hi - lo) if hi > lo else np.zeros_like(pc1)

    # 每个灰度档内 PC1 的跨度（序被压平的程度）
    bins = np.rint(gray).astype(np.int32)
    occ = np.unique(bins)
    spreads = []
    for b in occ:
        m = bins == b
        if m.sum() >= 5:
            spreads.append(float(pc1n[m].max() - pc1n[m].min()))
    spreads = np.array(spreads) if spreads else np.array([0.0])

    return {
        "file": os.path.basename(path),
        "r_eq_g_eq_b": bool(np.array_equal(flat[:, 0], flat[:, 1])
                            and np.array_equal(flat[:, 1], flat[:, 2])),
        "uniq_color": int(np.unique(flat, axis=0).shape[0]),
        "uniq_gray": int(np.unique(np.rint(gray).astype(np.int32)).size),
        "pc1_var": pc1_var,
        "rho_gray": abs(_spearman(pc1, gray)),
        "rho_lum": abs(_spearman(pc1, lum)),
        "rho_r": abs(_spearman(pc1, r_ch)),
        "spread_mean": float(spreads.mean()),
        "spread_p90": float(np.percentile(spreads, 90)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="红外伪彩→灰度 信息损失审计（只读）")
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "Images", "infrared", "*")))
    rng = random.Random(args.seed)
    pick = rng.sample(files, min(args.sample, len(files)))

    rows = [r for r in (analyze_one(f, rng) for f in pick) if r]
    if not rows:
        raise SystemExit("[error] 无可用样本")

    pc1 = np.array([r["pc1_var"] for r in rows])
    rg = np.array([r["rho_gray"] for r in rows])
    rl = np.array([r["rho_lum"] for r in rows])
    rr = np.array([r["rho_r"] for r in rows])
    sm = np.array([r["spread_mean"] for r in rows])
    sp90 = np.array([r["spread_p90"] for r in rows])
    gray_stack = sum(1 for r in rows if r["r_eq_g_eq_b"])

    print("=" * 78)
    print(f"红外渲染审计（随机 {len(rows)} 张 / 共 {len(files)} 张）")
    print("=" * 78)
    print(f"  真灰度堆叠 R==G==B          {gray_stack} / {len(rows)}"
          f"   {'→ 转灰度无损' if gray_stack == len(rows) else '→ 存在颜色信息'}")
    print(f"  PC1 方差占比（1.0 = 纯伪彩曲线）")
    print(f"      median {np.median(pc1):.3f}   p10 {np.percentile(pc1, 10):.3f}"
          f"   >=0.9 的图 {int((pc1 >= 0.9).sum())} / {len(rows)}")
    print(f"  颜色 → 灰度 的压缩")
    print(f"      唯一颜色数 median {np.median([r['uniq_color'] for r in rows]):.0f}"
          f"   唯一灰度数 median {np.median([r['uniq_gray'] for r in rows]):.0f}")
    print()
    print("  ★ 灰度是否保序（与温度代理 PC1 的 |Spearman|，1.0 = 完全保序）")
    print(f"      gray=mean(RGB)   median {np.median(rg):.3f}   p10 {np.percentile(rg, 10):.3f}")
    print(f"      luma 0.299/0.587/0.114  median {np.median(rl):.3f}")
    print(f"      R 通道           median {np.median(rr):.3f}")
    print()
    print("  ★ 同一灰度档内的温度跨度（相对全图，0 = 每个灰度唯一温度）")
    print(f"      mean spread  median {np.median(sm):.3f}   p90 {np.percentile(sm, 90):.3f}")
    print(f"      worst-bin spread  median {np.median(sp90):.3f}")

    worst = sorted(rows, key=lambda r: r["rho_gray"])[:5]
    print("\n  保序最差的 5 张（rho_gray 最小 → 灰度最能误导温度判断）：")
    for r in worst:
        print(f"      {r['file']}  rho_gray={r['rho_gray']:.3f}  "
              f"pc1_var={r['pc1_var']:.3f}  spread={r['spread_mean']:.3f}")

    print("\n  读法：")
    print("    · 若 pc1_var 高、rho_gray 低、spread 大 → 红外是伪彩，转灰度把温度序压平了，")
    print("      应直接送彩色原图（或把伪彩反解回温度再渲染），而不是 mean(RGB)。")
    print("    · 若 rho_gray ≈ 1 且 spread ≈ 0 → 转灰度无损，此路无收益。")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"sampled": len(rows), "files": len(files),
                       "summary": {
                           "gray_stack": gray_stack,
                           "pc1_var_median": float(np.median(pc1)),
                           "pc1_var_ge_0.9": int((pc1 >= 0.9).sum()),
                           "rho_gray_median": float(np.median(rg)),
                           "rho_lum_median": float(np.median(rl)),
                           "rho_r_median": float(np.median(rr)),
                           "spread_mean_median": float(np.median(sm)),
                           "spread_p90_median": float(np.median(sp90)),
                       },
                       "rows": rows}, fh, ensure_ascii=False, indent=1)
        print(f"\n已导出：{args.json}")


if __name__ == "__main__":
    main()