from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def fig_query_len(wlens, out):
    a = np.array(wlens)
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    bins = np.arange(0, a.max() + 3, 1)
    ax.hist(a, bins=bins, color="#5B8DEF", edgecolor="white", alpha=0.9)
    ax.axvline(np.median(a), color="#E06C00", ls="--", lw=1.4, label=f"中位 {np.median(a):.0f}")
    ax.axvline(np.mean(a), color="#2E8B57", ls="--", lw=1.4, label=f"均值 {np.mean(a):.2f}")
    ax.set_xlabel("查询词数")
    ax.set_ylabel("查询条数")
    ax.set_title("查询文本长度分布（词数）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_scene_queries(per, out):
    keys = sorted(int(k) for k in per)
    vals = [per[str(k)] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.bar(keys, vals, color="#5B8DEF", edgecolor="white")
    for x, v in zip(keys, vals):
        ax.text(x, v, str(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_xlabel("单场景查询数")
    ax.set_ylabel("场景数")
    ax.set_title("每场景查询数分布")
    ax.set_xticks(keys)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_depth_invalid(ratios, out):
    a = np.array(ratios) * 100
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.hist(a, bins=50, color="#8E6FC9", edgecolor="white", alpha=0.9)
    ax.axvline(np.median(a), color="#E06C00", ls="--", lw=1.4, label=f"中位 {np.median(a):.1f}%")
    ax.set_xlabel("深度无效像素占比（%）")
    ax.set_ylabel("场景数")
    ax.set_title("深度图无效像素（==0）占比分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_brightness(vm, im, out):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    ax.hist(vm, bins=60, alpha=0.55, color="#D9822B", label=f"可见光（中位 {np.median(vm):.1f}）")
    ax.hist(im, bins=60, alpha=0.55, color="#4E8A6F", label=f"红外（中位 {np.median(im):.1f}）")
    ax.set_xlabel("整图灰度均值")
    ax.set_ylabel("图像数")
    ax.set_title("可见光与红外亮度分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True, help="analyze_dataset.py 导出的统计 JSON")
    ap.add_argument("--outdir", required=True, help="配图输出目录")
    args = ap.parse_args()

    with open(args.stats, encoding="utf-8") as f:
        s = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    fig_query_len(s["query_word_lens"], os.path.join(args.outdir, "fig_query_len.png"))
    fig_scene_queries(s["per_scene"], os.path.join(args.outdir, "fig_scene_queries.png"))
    fig_depth_invalid(s["depth_invalid"], os.path.join(args.outdir, "fig_depth_invalid.png"))
    fig_brightness(s["visible_brightness"], s["infrared_brightness"], os.path.join(args.outdir, "fig_brightness.png"))
    print("figs written to", args.outdir)


if __name__ == "__main__":
    main()
