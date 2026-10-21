"""container 桶典型错误样本可视化（极端度排序版）。

排序：oversized 按 预测面积/真值面积 降序；undersized 按 真值/预测 降序；
misplaced 按中心距降序。pred=红粗线+PRED 标签，gt=绿粗线+GT 标签。

用法：
    python scripts/初赛/draw_container_cases.py --diag <diag.json> --queries dataset/queries/queries.json \
        --data-root dataset --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from PIL import Image, ImageDraw

from src.utils.io import load_json

PRED_COLOR = (255, 60, 60)
GT_COLOR = (40, 220, 120)


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center_dist(a: list[float], b: list[float]) -> float:
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--queries", default="dataset/queries/queries.json")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=6)
    args = ap.parse_args()

    diag = load_json(args.diag)
    queries = load_json(args.queries)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def pick(kind: str) -> list[dict]:
        items = diag["buckets"].get(kind, {}).get("top", [])
        if kind == "oversized":
            return sorted(items, key=lambda x: area(x["pred"]) / max(1e-9, area(x["gt"])), reverse=True)
        if kind == "undersized":
            return sorted(items, key=lambda x: area(x["gt"]) / max(1e-9, area(x["pred"])), reverse=True)
        return sorted(items, key=lambda x: center_dist(x["pred"], x["gt"]), reverse=True)

    drawn = 0
    for kind in ("oversized", "undersized", "misplaced"):
        for it in pick(kind)[: args.per_kind]:
            qid = it["qid"]
            meta = queries[qid]
            vis_path = os.path.join(args.data_root, meta["visible"])
            if not os.path.exists(vis_path):
                print("missing:", vis_path)
                continue
            img = Image.open(vis_path).convert("RGB")
            d = ImageDraw.Draw(img)
            w, h = img.size
            for label, box, color in (("PRED", it["pred"], PRED_COLOR), ("GT", it["gt"], GT_COLOR)):
                if not box:
                    continue
                x1, y1, x2, y2 = [v * (w if i % 2 == 0 else h) for i, v in enumerate(box)]
                d.rectangle([x1, y1, x2, y2], outline=color, width=6)
                d.text((x1, max(0, y1 - 22)), label, fill=color)
            fn = out_dir / f"{kind}_{qid}.png"
            img.save(fn)
            drawn += 1
    print("drawn:", drawn, "->", out_dir)


if __name__ == "__main__":
    main()
