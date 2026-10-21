# -*- coding: utf-8 -*-
"""独立复核脚本：重新计算文档《数据集说明.md》声明的全部关键数字，输出对照结果。

用法: .venv\\Scripts\\python.exe scripts/verify_doc.py [--root dataset] [--output outputs/verify_doc.json]
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def main() -> None:
    ap = argparse.ArgumentParser(description="复核数据集说明文档中的关键统计数字")
    ap.add_argument("--root", default="dataset", help="数据集目录名（相对项目根目录）")
    ap.add_argument("--output", default=None, help="统计结果 JSON 输出路径（默认 outputs/verify_doc.json）")
    args = ap.parse_args()

    base = os.path.join(PROJECT_ROOT, args.root)
    img_dir = os.path.join(base, "Images")
    qpath = os.path.join(base, "queries", "queries.json")

    with open(qpath, encoding="utf-8") as f:
        q = json.load(f)

    res = {}

    # ---------- 1. queries.json 结构 ----------
    res["queries_total"] = len(q)
    res["keys_unique"] = len(set(q))
    res["key_format_ok"] = all(re.fullmatch(r"\d{6}_\d{3}", k) for k in q)
    res["fields"] = sorted(set().union(*(v.keys() for v in q.values())))
    res["has_bbox"] = sum("bbox" in v for v in q.values())
    res["empty_query"] = sum(not v["query"].strip() for v in q.values())
    res["missing_files"] = sum(1 for v in q.values() for k in ("visible", "infrared", "depth")
                               if not os.path.isfile(os.path.join(base, v[k])))
    # 孤儿文件
    dir_ids = {}
    for mod in ("visible", "infrared", "depth"):
        dir_ids[mod] = {os.path.basename(p) for p in glob.glob(os.path.join(img_dir, mod, "*.*"))}
    ref_ids = {mod: {os.path.basename(v[mod]) for v in q.values()} for mod in dir_ids}
    res["orphan_files"] = {mod: len(dir_ids[mod] - ref_ids[mod]) for mod in dir_ids}

    # ---------- 2. 文件格式/尺寸 ----------
    fmt = {}
    for mod in ("visible", "infrared", "depth"):
        c = Counter()
        for fp in glob.glob(os.path.join(img_dir, mod, "*.*")):
            with Image.open(fp) as im:
                c[(im.format, im.mode, im.size)] += 1
        fmt[mod] = {str(k): v for k, v in c.items()}
    res["fmt"] = fmt

    # ---------- 3. 深度图（全量 PNG） ----------
    dratio, dmax_vals = [], []
    for fp in glob.glob(os.path.join(img_dir, "depth", "*.png")):
        a = np.array(Image.open(fp))
        dratio.append((a == 0).mean())
        dmax_vals.append(int(a.max()))
    dratio = np.array(dratio)
    res["depth_invalid_mean"] = round(float(dratio.mean()), 4)
    res["depth_invalid_median"] = round(float(np.median(dratio)), 4)
    res["depth_invalid_max"] = round(float(dratio.max()), 4)
    res["depth_gt90"] = int((dratio > 0.9).sum())
    res["depth_gt50"] = int((dratio > 0.5).sum())
    res["depth_lt10"] = int((dratio < 0.1).sum())
    res["depth_max_all"] = (max(dmax_vals), min(dmax_vals))
    worst_id = None
    for fp, r in zip(glob.glob(os.path.join(img_dir, "depth", "*.png")), dratio):
        if worst_id is None or r > worst_id[1]:
            worst_id = (os.path.basename(fp), float(r))
    res["depth_worst"] = worst_id

    # ---------- 4. 亮度（全量 2000） ----------
    vm, ime = [], []
    for fp in sorted(glob.glob(os.path.join(img_dir, "visible", "*.*"))):
        vm.append(float(np.array(Image.open(fp).convert("L")).mean()))
    for fp in sorted(glob.glob(os.path.join(img_dir, "infrared", "*.*"))):
        ime.append(float(np.array(Image.open(fp).convert("L")).mean()))
    vm, ime = np.array(vm), np.array(ime)
    res["vis_median"] = round(float(np.median(vm)), 1)
    res["vis_p5"] = round(float(np.percentile(vm, 5)), 1)
    res["vis_p95"] = round(float(np.percentile(vm, 95)), 1)
    res["vis_dark_n"] = int((vm < 60).sum())
    dark_ids = [os.path.basename(fp) for fp, v in zip(sorted(glob.glob(os.path.join(img_dir, "visible", "*.*"))), vm) if v < 60]
    res["vis_dark_ids"] = sorted(dark_ids)
    res["ir_median"] = round(float(np.median(ime)), 1)
    res["ir_p5"] = round(float(np.percentile(ime, 5)), 1)
    res["ir_p95"] = round(float(np.percentile(ime, 95)), 1)

    # ---------- 5. 查询长度 ----------
    wlens = [len(v["query"].split()) for v in q.values()]
    clens = [len(v["query"]) for v in q.values()]
    res["wl_min"], res["wl_max"] = min(wlens), max(wlens)
    res["wl_mean"] = round(float(np.mean(wlens)), 2)
    res["wl_median"] = int(np.median(wlens))
    res["wl_p90"] = int(np.percentile(wlens, 90))
    res["wl_p95"] = int(np.percentile(wlens, 95))
    res["cl_min"], res["cl_max"] = min(clens), max(clens)
    res["cl_mean"] = round(float(np.mean(clens)), 1)
    res["wl_gt30"] = sum(1 for w in wlens if w > 30)

    # ---------- 6. 每场景查询数 ----------
    per = Counter(k.split("_")[0] for k in q)
    res["scene_n"] = len(per)
    res["per_scene"] = dict(sorted(Counter(per.values()).items()))

    # ---------- 7. 词频 ----------
    words = [w.lower() for ws in (v["query"].split() for v in q.values()) for w in ws]
    wc = Counter(words)
    res["top10"] = wc.most_common(10)
    for k in ["white", "black", "blue", "red", "green", "yellow", "dark", "light", "gray", "brown", "pink"]:
        res[f"w_{k}"] = wc.get(k, 0)
    res["left"] = wc.get("left", 0)
    res["leftmost"] = wc.get("leftmost", 0)
    res["right"] = wc.get("right", 0)
    res["rightmost"] = wc.get("rightmost", 0)
    for k in ["far", "behind", "near", "front", "top", "center", "middle", "first", "second", "closest",
              "standing", "wearing", "walking", "sitting", "facing", "holding", "riding", "parked",
              "two", "one", "count", "three", "person", "camera", "sign"]:
        res[f"w_{k}"] = wc.get(k, 0)

    # ---------- 8. 语言混合 ----------
    cjk = re.compile(r"[\u4e00-\u9fff]")
    pure_cn, mixed_cn = [], []
    for k, v in q.items():
        if cjk.search(v["query"]):
            (pure_cn if not re.findall(r"[A-Za-z]", v["query"]) else mixed_cn).append((k, v["query"]))
    res["cjk_total"] = len(pure_cn) + len(mixed_cn)
    res["pure_cn"] = [k for k, _ in pure_cn]
    res["mixed_cn_n"] = len(mixed_cn)
    bom_q = [(k, v["query"]) for k, v in q.items() if "\ufeff" in v["query"]]
    hyphen_q = [(k, v["query"]) for k, v in q.items() if "\u2011" in v["query"]]
    res["bom_n"] = len(bom_q)
    res["bom_ids"] = [k for k, _ in bom_q]
    res["hyphen_n"] = len(hyphen_q)
    res["hyphen_ids"] = [k for k, _ in hyphen_q]
    res["pure_ascii"] = sum(1 for v in q.values() if all(c.isascii() or c == " " for c in v["query"]))

    # ---------- 9. 同场景重复文本 ----------
    by_scene = {}
    for k, v in q.items():
        by_scene.setdefault(k.split("_")[0], []).append(v["query"])
    res["dup_scene_n"] = sum(1 for qs in by_scene.values() if len(set(qs)) != len(qs))

    # ---------- 10. 红外通道差异（全量 PNG 扫描 R-B 最大差 与 R-G>3 占比） ----------
    rb_max = 0
    rg_big_ratio_sum = 0.0
    n_scanned = 0
    for fp in glob.glob(os.path.join(img_dir, "infrared", "*.png")):
        a = np.array(Image.open(fp))
        rb = np.abs(a[..., 0].astype(np.int16) - a[..., 2].astype(np.int16)).max()
        rg_big = (np.abs(a[..., 0].astype(np.int16) - a[..., 1].astype(np.int16)) > 3).mean()
        rb_max = max(rb_max, int(rb))
        rg_big_ratio_sum += rg_big
        n_scanned += 1
    res["ir_rb_max"] = rb_max
    res["ir_rg_big_ratio"] = round(rg_big_ratio_sum / n_scanned, 6)
    res["ir_scanned"] = n_scanned

    # ---------- 11. ID 规律 ----------
    vis_files = [os.path.basename(p) for p in glob.glob(os.path.join(img_dir, "visible", "*.*"))]
    std_files = [f for f in vis_files if re.fullmatch(r"\d{6}\.\w+", f)]
    nonstd = sorted(set(vis_files) - set(std_files))
    res["nonstd_files"] = nonstd
    ids = sorted(int(os.path.splitext(p)[0]) for p in std_files)
    diffs = np.diff(ids)
    res["id_min"], res["id_max"], res["id_n"] = ids[0], ids[-1], len(ids)
    res["gap7_n"] = int((diffs == 7).sum())
    res["gap_total"] = len(diffs)
    jpg_ids = sorted(int(os.path.splitext(p)[0]) for p in std_files if p.endswith(".jpg"))
    res["jpg_n"] = len(jpg_ids)
    res["jpg_first"], res["jpg_last"] = jpg_ids[0], jpg_ids[-1]
    res["jpg_contig"] = all(b - a == 7 for a, b in zip(jpg_ids, jpg_ids[1:]))
    res["jpg_query_n"] = sum(1 for k in q if ".jpg" in q[k]["visible"])
    jpg_txt = " ".join(q[k]["query"] for k in q if ".jpg" in q[k]["visible"])
    jpg_wc = Counter(w.lower() for w in jpg_txt.split())
    res["jpg_drone"] = jpg_wc.get("drone", 0)
    res["jpg_flying"] = jpg_wc.get("flying", 0)

    # ---------- 12. 体积（字节 → MiB / GiB / GB） ----------
    total_bytes = 0
    for mod in ("visible", "infrared", "depth"):
        b = sum(os.path.getsize(p) for p in glob.glob(os.path.join(img_dir, mod, "*.*")))
        total_bytes += b
        res[f"size_{mod}_MiB"] = round(b / (1024 * 1024), 2)
        res[f"size_{mod}_GiB"] = round(b / (1024 ** 3), 2)
        res[f"size_{mod}_GB"] = round(b / 1e9, 2)
    res["size_total_MiB"] = round(total_bytes / (1024 * 1024), 2)
    res["size_total_GiB"] = round(total_bytes / (1024 ** 3), 2)
    res["size_total_GB"] = round(total_bytes / 1e9, 2)

    # ---------- 13. 扩展名与实际解码格式不一致的 PNG ----------
    mismatched = []
    for fp in sorted(glob.glob(os.path.join(img_dir, "visible", "*.png"))):
        with Image.open(fp) as im:
            if im.format != "PNG":
                mismatched.append((os.path.basename(fp), im.format, im.size))
    res["png_format_mismatch"] = mismatched

    out_path = args.output or os.path.join(PROJECT_ROOT, "outputs", "verify_doc.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, default=str)

    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n统计已导出: {out_path}")


if __name__ == "__main__":
    main()
