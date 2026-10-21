# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash_bytes(img: Image.Image) -> np.ndarray:
    gray = img.convert("L").resize((9, 8), Image.LANCZOS)
    a = np.asarray(gray, dtype=np.uint16)
    bits = (a[:, 1:] < a[:, :-1]).reshape(-1)
    return np.packbits(bits)


def histogram_feature(img: Image.Image, bins: int = 32) -> np.ndarray:
    gray = img.convert("L").resize((128, 72), Image.LANCZOS)
    h, _ = np.histogram(np.asarray(gray, dtype=np.uint8).ravel(), bins=bins, range=(0, 256))
    return h.astype(np.float32) / max(1, int(h.sum()))


def hist_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt(float((a0 * a0).sum()) * float((b0 * b0).sum()))
    if denom == 0:
        return 0.0
    return float((a0 * b0).sum() / denom)


def hamming_matrix(digests: list[np.ndarray]) -> np.ndarray:
    x = np.asarray(digests, dtype=np.uint8)
    return POPCOUNT[x[:, None, :] ^ x[None, :, :]].sum(axis=2).astype(np.uint16)


def rank_gap_bucket(gap: int) -> str:
    if gap <= 2:
        return "1-2"
    if gap <= 4:
        return "3-4"
    if gap <= 8:
        return "5-8"
    if gap <= 16:
        return "9-16"
    return "17+"


def similar_clusters(pairs: list[tuple[int, int]], n: int) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="数据集根目录（相对项目根目录）")
    ap.add_argument("--similarity-json", default=None, help="scene_similarity JSON（用于连通分量）")
    ap.add_argument("--json", default=None, help="导出统计 JSON 的路径（可选）")
    args = ap.parse_args()

    base = os.path.abspath(args.root)
    img_dir = os.path.join(base, "Images")

    jpg_files = sorted(f for f in os.listdir(os.path.join(img_dir, "visible"))
                       if f.lower().endswith(".jpg"))
    ids = [os.path.splitext(f)[0] for f in jpg_files]
    print(f"JPG 场景数: {len(ids)}")

    stats: dict = {}
    mod_stats = {}
    for mod in ("visible", "infrared", "depth"):
        digests = []
        hists = []
        for sid in ids:
            fp = os.path.join(img_dir, mod, f"{sid}.jpg")
            if not os.path.isfile(fp):
                fp = os.path.join(img_dir, mod, f"{sid}.png")
            with Image.open(fp) as im:
                digests.append(dhash_bytes(im))
                hists.append(histogram_feature(im))
        mat = hamming_matrix(digests)
        n = mat.shape[0]
        rank = np.arange(n)
        gaps = np.abs(rank[:, None] - rank[None, :])
        up = np.triu_indices(n, 1)
        g = gaps[up]
        v = mat[up]
        buckets = {}
        for b in sorted(set(rank_gap_bucket(x) for x in g)):
            mask = np.array([rank_gap_bucket(x) == b for x in g])
            buckets[b] = {
                "count": int(mask.sum()),
                "mean_h": float(v[mask].mean()),
                "median_h": float(np.median(v[mask])),
                "le5": int((v[mask] <= 5).sum()),
                "le10": int((v[mask] <= 10).sum()),
            }
        mod_stats[mod] = {"buckets": buckets, "all_mean": float(v.mean()),
                          "all_median": float(np.median(v))}
        print(f"[{mod}] 全对: mean={v.mean():.2f} median={np.median(v):.1f}")
        for b, bv in buckets.items():
            print(f"   帧间隔 {b}: count={bv['count']} mean={bv['mean_h']:.2f} "
                  f"median={bv['median_h']:.1f} le5={bv['le5']} le10={bv['le10']}")
    stats["jpg_modality"] = mod_stats

    if args.similarity_json and os.path.isfile(args.similarity_json):
        vis_files = sorted(f for f in os.listdir(os.path.join(img_dir, "visible"))
                           if f.lower().endswith((".png", ".jpg", ".jpeg")))
        id_to_idx = {os.path.splitext(f)[0]: i for i, f in enumerate(vis_files)}
        with open(args.similarity_json, encoding="utf-8") as f:
            sim = json.load(f)
        pairs = [(id_to_idx[p["id_i"]], id_to_idx[p["id_j"]])
                 for p in sim.get("near5_pairs", [])
                 if p["id_i"] in id_to_idx and p["id_j"] in id_to_idx]
        clusters = similar_clusters(pairs, len(vis_files))
        sizes = [len(c) for c in clusters]
        stats["near5_clusters"] = {
            "n_clusters": len(clusters),
            "size_dist": {str(s): sizes.count(s) for s in sorted(set(sizes))},
            "max_size": max(sizes) if sizes else 0,
        }
        print("near5 连通分量:", stats["near5_clusters"])
        big = [c for c in clusters if len(c) >= 4]
        stats["near5_cluster_examples"] = big[:5]
        print("≥4 张的簇示例:", big[:5])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"\n已导出: {args.json}")


if __name__ == "__main__":
    main()
