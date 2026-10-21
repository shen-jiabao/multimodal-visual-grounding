# -*- coding: utf-8 -*-
"""数据集统计探查脚本（只读，不修改任何数据）。

用法（项目根目录下）:
    .venv\\Scripts\\python.exe scripts/analyze_dataset.py [--root dataset] [--json out.json]

输出:
    - 控制台打印关键统计摘要
    - 可选 --json 将原始统计（查询词长、每场景查询数、深度无效率、亮度等）导出为 JSON，
      供文档配图 / 后续分析复用。

说明:
    - 本脚本不依赖 Grounding DINO 等模型，仅用 Pillow + numpy 做文件级统计。
    - 全量统计约需 1~2 分钟（含 1903 张 uint16 深度图读取）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
from PIL import Image


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

    with open(qpath, encoding="utf-8") as f:
        q = json.load(f)

    stats: dict = {}
    print("==== 1. queries.json 校验 ====")
    print(f"总查询数: {len(q)}")
    print(f"键去重: {len(set(q))} | 字段: {sorted(set().union(*(v.keys() for v in q.values())))}")
    print(f"含 bbox 条目: {sum('bbox' in v for v in q.values())}")
    print(f"空查询: {sum(not v['query'].strip() for v in q.values())}")
    cjk = re.compile(r'[\u4e00-\u9fff]')
    print(f"含中文查询: {sum(bool(cjk.search(v['query'])) for v in q.values())}")
    ascii_n = sum(1 for v in q.values() if all(c.isascii() or c == ' ' for c in v['query']))
    print(f"纯 ASCII 查询: {ascii_n} / {len(q)}")

    print("\n==== 2. 文件存在性 ====")
    missing = sum(1 for v in q.values() for k in ("visible", "infrared", "depth")
                  if not os.path.isfile(os.path.join(base, v[k])))
    print(f"引用但缺失的文件: {missing}")

    print("\n==== 3. 图像格式（全量） ====")
    fmt_summary = {}
    for mod in ("visible", "infrared", "depth"):
        fmt = Counter()
        for fp in glob.glob(os.path.join(img_dir, mod, "*.*")):
            with Image.open(fp) as im:
                fmt[(im.mode, im.size, im.format)] += 1
        fmt_summary[mod] = {f"{k}": v for k, v in fmt.items()}
        print(f"{mod}: {dict(fmt)}")

    print("\n==== 4. 深度图统计（PNG，uint16） ====")
    dratio, dmax = [], []
    for fp in glob.glob(os.path.join(img_dir, "depth", "*.png")):
        a = np.array(Image.open(fp))
        dratio.append((a == 0).mean())
        dmax.append(a.max())
    dratio = np.array(dratio)
    print(f"深度无效(==0)占比: 均值={dratio.mean()*100:.2f}% 中位={np.median(dratio)*100:.2f}% "
          f"最大={dratio.max()*100:.2f}% >50%:{(dratio > 0.5).sum()} >90%:{(dratio > 0.9).sum()}")
    stats["depth_invalid"] = dratio.tolist()

    print("\n==== 5. 亮度分布（全量 2000 场景） ====")
    vmean, imean = [], []
    for fp in glob.glob(os.path.join(img_dir, "visible", "*.*")):
        vmean.append(float(np.array(Image.open(fp).convert("L")).mean()))
    for fp in glob.glob(os.path.join(img_dir, "infrared", "*.*")):
        imean.append(float(np.array(Image.open(fp).convert("L")).mean()))
    print(f"可见光均值: 中位={np.median(vmean):.1f} 5分位={np.percentile(vmean,5):.1f} "
          f"95分位={np.percentile(vmean,95):.1f} 暗(<60)={(np.array(vmean) < 60).sum()}")
    print(f"红外均值: 中位={np.median(imean):.1f} 5分位={np.percentile(imean,5):.1f} "
          f"95分位={np.percentile(imean,95):.1f}")
    stats["visible_brightness"] = vmean
    stats["infrared_brightness"] = imean

    print("\n==== 6. 查询文本长度 ====")
    wlens = [len(v["query"].split()) for v in q.values()]
    print(f"词数: min={min(wlens)} max={max(wlens)} 均值={np.mean(wlens):.2f} "
          f"中位={np.median(wlens):.0f} 90分位={np.percentile(wlens,90):.0f}")
    stats["query_word_lens"] = wlens

    print("\n==== 7. 每场景查询数 ====")
    per = Counter(k.split("_")[0] for k in q)
    print(f"场景数: {len(per)} | 分布: {dict(sorted(Counter(per.values()).items()))}")
    stats["per_scene"] = {str(k): v for k, v in sorted(Counter(per.values()).items())}

    print("\n==== 8. JPG 低分辨率区块 ====")
    jpg_ids = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(img_dir, "visible", "*.jpg")))
    jpg_q = [k for k in q if ".jpg" in q[k]["visible"]]
    print(f"JPG 场景: {len(jpg_ids)} 个（{jpg_ids[0] if jpg_ids else '-'} ~ {jpg_ids[-1] if jpg_ids else '-'}），"
          f"关联查询 {len(jpg_q)} 条")
    stats["jpg_ids"] = jpg_ids
    stats["jpg_query_n"] = len(jpg_q)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False)
        print(f"\n统计已导出: {args.json}")


if __name__ == "__main__":
    main()
