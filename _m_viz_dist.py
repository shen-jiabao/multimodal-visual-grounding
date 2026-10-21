#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M 件低分证据的数字剖面图（离线，一张 PNG）。

与 `_m_viz_split.py` 同源数据：M 件 vs 基线（0.7963），改动 1330 条
按 序数(484) / 非序数(846) 分层，画两件事：
  ① 改动条目 IoU(基线, M) 的分布直方图 —— 看"判到不同区域"有多极端
  ② 面积比 (M/基线) 的分布 —— 排除"框整体变大/变小"的系统性换算错
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal"
M = SUB / "未提交" / "result_semi_m3m_rules5_mb.json"
BEST = SUB / "0.7963" / "result_semi_q38_nonord.json"
CORES = R / "_ordinal_cores_r2.json"
OUT = R / "m_viz_dist.png"


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if (ar + br - inter) > 0 else 0.0


def _area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def main() -> int:
    m = json.loads(M.read_text(encoding="utf-8"))
    b = json.loads(BEST.read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))

    layers = {"ordinal 484": [], "non-ordinal 846": []}
    for k in m:
        if m[k]["bbox"] == b[k]["bbox"]:
            continue
        tag = "ordinal 484" if k in cores else "non-ordinal 846"
        io = _iou(b[k]["bbox"], m[k]["bbox"])
        ab, am = _area(b[k]["bbox"]), _area(m[k]["bbox"])
        ratio = am / ab if ab > 0 else float("nan")
        layers[tag].append((io, ratio))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"ordinal 484": "tab:red", "non-ordinal 846": "tab:blue"}

    # ① IoU 直方图（线性刻度）
    ax = axes[0][0]
    for tag, pts in layers.items():
        ax.hist([p[0] for p in pts], bins=20, range=(0, 1), alpha=0.55,
                label=f"{tag}  IoU<0.3: "
                      f"{sum(1 for q,_ in pts if q < 0.3)/len(pts):.0%}",
                color=colors[tag])
    ax.axvline(0.5, color="k", ls="--", lw=1)
    ax.text(0.51, ax.get_ylim()[1] * 0.9, "ACC@0.5", fontsize=9)
    ax.set_title("Changed entries: IoU vs baseline (linear)")
    ax.legend(fontsize=9)

    # ② IoU 直方图（对数刻度，突出 IoU=0 的质量）
    ax = axes[0][1]
    for tag, pts in layers.items():
        ax.hist([p[0] for p in pts], bins=20, range=(0, 1),
                weights=None, alpha=0.55, color=colors[tag], log=True,
                label=tag)
    ax.set_title("Same, log y-axis (mass at IoU=0 dominates)")
    ax.legend(fontsize=9)

    # ③ 面积比分布
    ax = axes[1][0]
    for tag, pts in layers.items():
        rs = [p[1] for p in pts if p[1] == p[1]]
        ax.hist(rs, bins=40, range=(0, 3), alpha=0.55, color=colors[tag],
                label=f"{tag}  p50={sorted(rs)[len(rs)//2]:.2f}")
    ax.axvline(1.0, color="k", ls="--", lw=1)
    ax.set_title("Area ratio  M/baseline  (1.0 = unchanged size)")
    ax.legend(fontsize=9)

    # ④ 汇总文字（与 _m_why.py 口径一致）
    ax = axes[1][1]
    ax.axis("off")
    lines = []
    for tag, pts in layers.items():
        ios = sorted(p[0] for p in pts)
        n = len(ios)
        lines += [
            f"{tag}   (n={n})",
            f"  IoU p10/p50/p90 = {ios[n//10]:.3f} / "
            f"{ios[n//2]:.3f} / {ios[9*n//10]:.3f}",
            f"  IoU<0.3: {sum(1 for q in ios if q < 0.3)/n:.1%}"
            f"   IoU=0: {sum(1 for q in ios if q == 0)/n:.1%}",
            f"  IoU>=0.5: {sum(1 for q in ios if q >= 0.5)/n:.1%}",
            "",
        ]
    lines += [
        "Reading:",
        "  median IoU 0 + 80%<0.3 => M judged DIFFERENT regions,",
        "  not a coordinate-scaling bug (that would shift boxes",
        "  together and keep high IoU).",
        "  area ratio p50~1 => no systematic shrink/expand.",
    ]
    ax.text(0.02, 0.95, "\n".join(lines), fontsize=11, family="monospace",
            va="top")

    fig.suptitle("M (0.7439, -298) vs baseline q38 (0.7963) — "
                 "1330 changed entries split by layer")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT, dpi=110)
    print(f"已写 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
