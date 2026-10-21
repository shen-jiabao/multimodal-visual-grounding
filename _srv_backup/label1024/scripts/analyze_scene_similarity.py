# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
from PIL import Image

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
NEAR_HAMMING = 12
SAMPLES = 10000


def load_visible_files(img_dir: str) -> list[str]:
    files = sorted(os.listdir(os.path.join(img_dir, "visible")))
    return [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]


def scene_id_from_filename(name: str) -> str:
    m = re.match(r"(\d{6})", name)
    return m.group(1) if m else name


def load_referenced_ids(qpath: str) -> set[str]:
    with open(qpath, encoding="utf-8") as f:
        q = json.load(f)
    return {os.path.splitext(os.path.basename(v["visible"]))[0] for v in q.values()}


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


def gap_bucket(gap: int) -> str:
    if gap <= 2:
        return "1-2"
    if gap <= 5:
        return "3-5"
    if gap <= 10:
        return "6-10"
    if gap <= 50:
        return "11-50"
    if gap <= 200:
        return "51-200"
    return ">200"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="数据集根目录（相对项目根目录）")
    ap.add_argument("--json", default=None, help="导出统计 JSON 的路径（可选）")
    args = ap.parse_args()

    base = os.path.abspath(args.root)
    img_dir = os.path.join(base, "Images")
    qpath = os.path.join(base, "queries", "queries.json")
    assert os.path.isdir(img_dir), f"未找到图像目录: {img_dir}"
    assert os.path.isfile(qpath), f"未找到查询文件: {qpath}"

    files = load_visible_files(img_dir)
    ids = [scene_id_from_filename(f) for f in files]
    referenced = load_referenced_ids(qpath)
    ref_idx = np.array([i for i, sid in enumerate(ids) if sid in referenced], dtype=np.int64)
    id_vals = np.array([int(sid) for sid in ids], dtype=np.int64)

    print(f"visible 文件数: {len(files)} | 被查询引用: {len(ref_idx)}")

    digests = []
    hists = {}
    for i, f in enumerate(files):
        with Image.open(os.path.join(img_dir, "visible", f)) as im:
            digests.append(dhash_bytes(im))
            hists[i] = histogram_feature(im)
    mat = hamming_matrix(digests)
    n = mat.shape[0]

    diag_mask = np.eye(n, dtype=bool)
    pair_vals = mat[~diag_mask]
    pair_ids = np.argwhere(~diag_mask)
    gaps = np.abs(id_vals[pair_ids[:, 0]] - id_vals[pair_ids[:, 1]])

    stats: dict = {}
    stats["n_images"] = int(n)
    stats["n_referenced"] = int(len(ref_idx))

    dist_all, _ = np.histogram(pair_vals, bins=64, range=(0, 64))
    stats["hamming_dist_all"] = dist_all.astype(int).tolist()

    bucket_stats = {}
    for gap in sorted(set(gap_bucket(g) for g in gaps)):
        mask = np.array([gap_bucket(g) == gap for g in gaps])
        bucket_stats[gap] = {
            "count": int(mask.sum()),
            "mean": float(pair_vals[mask].mean()),
            "p10": float(np.percentile(pair_vals[mask], 10)),
            "median": float(np.median(pair_vals[mask])),
            "p90": float(np.percentile(pair_vals[mask], 90)),
        }
    stats["id_gap_buckets"] = bucket_stats

    rng = np.random.default_rng(42)
    rand_pairs = rng.integers(0, n, size=(SAMPLES, 2))
    rand_vals = mat[rand_pairs[:, 0], rand_pairs[:, 1]]
    stats["random_baseline"] = {
        "mean": float(rand_vals.mean()),
        "median": float(np.median(rand_vals)),
        "p90": float(np.percentile(rand_vals, 90)),
    }

    ref_sub = mat[ref_idx]
    n_ref = ref_sub.shape[0]
    topk = np.argsort(ref_sub, axis=1)[:, 1:6]
    topk_vals = np.take_along_axis(ref_sub, topk, axis=1)
    stats["ref_topk_hamming"] = {
        f"top{k}": {
            "mean": float(topk_vals[:, k - 1].mean()),
            "median": float(np.median(topk_vals[:, k - 1])),
            "p10": float(np.percentile(topk_vals[:, k - 1], 10)),
            "p90": float(np.percentile(topk_vals[:, k - 1], 90)),
            "le10": int((topk_vals[:, k - 1] <= 10).sum()),
            "le16": int((topk_vals[:, k - 1] <= 16).sum()),
        }
        for k in range(1, 6)
    }
    stats["ref_with_near_support"] = {
        "le10": int((topk_vals[:, 0] <= 10).sum()),
        "le12": int((topk_vals[:, 0] <= NEAR_HAMMING).sum()),
        "le16": int((topk_vals[:, 0] <= 16).sum()),
    }

    near_pairs = np.argwhere((mat <= NEAR_HAMMING) & ~diag_mask)
    stats["near_pair_count"] = int(len(near_pairs))
    near_examples = []
    for i, j in near_pairs[:20]:
        near_examples.append({
            "id_i": ids[i],
            "id_j": ids[j],
            "hamming": int(mat[i, j]),
            "hist_corr": round(hist_correlation(hists[i], hists[j]), 4),
        })
    stats["near_pair_examples"] = near_examples

    dup_pairs = np.argwhere((mat <= 5) & ~diag_mask)
    stats["near5_pair_count"] = int(len(dup_pairs))
    stats["near5_pairs"] = [{"id_i": ids[i], "id_j": ids[j]} for i, j in dup_pairs]

    check_pairs = dup_pairs[:: max(1, len(dup_pairs) // 40)][:40]
    triplet_check = []
    for i, j in check_pairs:
        row = {"id_i": ids[i], "id_j": ids[j], "v_h": int(mat[i, j]),
               "v_c": round(hist_correlation(hists[i], hists[j]), 4)}
        for mod, key_h, key_c in (("infrared", "ir_h", "ir_c"), ("depth", "d_h", "d_c")):
            pa = os.path.join(img_dir, mod, f"{ids[i]}.png")
            pb = os.path.join(img_dir, mod, f"{ids[j]}.png")
            if os.path.isfile(pa) and os.path.isfile(pb):
                with Image.open(pa) as ia, Image.open(pb) as ib:
                    ha = dhash_bytes(ia)
                    hb = dhash_bytes(ib)
                    row[key_h] = int(POPCOUNT[ha ^ hb].sum())
                    row[key_c] = round(hist_correlation(histogram_feature(ia), histogram_feature(ib)), 4)
        triplet_check.append(row)
    stats["triplet_check"] = triplet_check

    for k, v in stats.items():
        if k in ("near_pair_examples", "near5_pairs", "triplet_check"):
            continue
        print(f"{k}: {json.dumps(v, ensure_ascii=False)[:200]}")

    print("near_pair_examples:")
    for ex in near_examples:
        print("  ", ex)

    print("triplet_check:")
    for row in triplet_check:
        print("  ", row)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"\n已导出: {args.json}")


if __name__ == "__main__":
    main()
