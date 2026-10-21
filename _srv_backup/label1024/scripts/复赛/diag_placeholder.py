# -*- coding: utf-8 -*-
"""占位框成因诊断：92 条中性占位框是在什么条件下产生的（只读）。

关键问题：中性占位框是"单条查询模型没给框"，还是"整张图生成崩掉（全图都占位）"？
两者修法完全不同（前者=单查询重试/换 prompt；后者=缩小同图批大小/加长 token/分批）。
"""
from __future__ import annotations

import json
import os
import sys
import tarfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PLACEHOLDER = [0.49, 0.49, 0.51, 0.51]
ARCHIVE = "outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz"


def main() -> None:
    with tarfile.open(ARCHIVE, "r:gz") as t:
        name = next(m.name for m in t.getmembers()
                    if os.path.basename(m.name) == "predictions.json")
        pred = json.load(t.extractfile(name))
    with open("dataset/queries/queries.json", encoding="utf-8") as f:
        q = json.load(f)

    vis = {}
    for k, v in q.items():
        vis.setdefault(v["visible"], []).append(k)

    ph_images = []
    for image, qids in vis.items():
        n = sum(1 for k in qids if [float(x) for x in pred[k]] == PLACEHOLDER)
        if n:
            ph_images.append((n, len(qids), image, qids))

    print("含占位框的图片数:", len(ph_images), "/", len(vis))
    print("占位框总数:", sum(n for n, _, _, _ in ph_images))
    print("占位图片中『全图皆占位』的图片数:",
          sum(1 for n, tot, _, _ in ph_images if n == tot))
    print("全图皆占位贡献的占位框数:",
          sum(n for n, tot, _, _ in ph_images if n == tot))
    print("按 (占位数, 该图查询数) 的分布:",
          dict(sorted(Counter((n, tot) for n, tot, _, _ in ph_images).items())))

    print("\n--- 占位最严重的 12 张图 ---")
    for n, tot, image, qids in sorted(ph_images, key=lambda r: (-r[1], r[0]))[:12]:
        print(f"  {os.path.basename(image)}  占位 {n}/{tot}")
        for k in qids:
            tag = "PLACEHOLDER" if [float(x) for x in pred[k]] == PLACEHOLDER else "ok"
            print(f"     [{tag:11s}] {k}  {q[k]['query'][:100]}")


if __name__ == "__main__":
    main()
