"""数据准备与校验脚本。

功能：
1. 校验数据集 JSON 格式与图像存在性。
2. 划分 train/val（如未提供 val.json）。
3. 简单可视化样本（保存带 bbox 的可见光图）。

用法：
    python prepare_data.py [--data_root <path>] [--val_ratio 0.1]
"""
from __future__ import annotations
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from config import (
    BBOX_KEY,
    DATA_ROOT,
    MODALITY_KEYS,
    TEXT_KEY,
    WORK_DIR,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--src_json", type=str, default="all.json",
                   help="包含全部样本的 JSON（若没有，将尝试合并 train.json+val.json）")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vis_out", type=str, default=str(WORK_DIR / "vis"))
    p.add_argument("--no_vis", action="store_true", help="不保存可视化图")
    return p.parse_args()


def load_all(data_root: Path) -> Dict[str, dict]:
    """加载数据根目录下的所有 JSON，合并为全集。"""
    candidates = []
    for name in ("all.json", "train.json", "sample.json"):
        p = data_root / name
        if p.exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"{data_root} 下未找到 all.json/train.json/sample.json")
    merged: Dict[str, dict] = {}
    for p in candidates:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        for k, v in d.items():
            merged[k] = v
    return merged


def validate(records: Dict[str, dict], data_root: Path) -> List[str]:
    """校验所有记录，统计并返回问题样本 ID 列表。"""
    problems = []
    for qid, rec in records.items():
        for mk, key in MODALITY_KEYS.items():
            p = data_root / rec[key]
            if not p.exists():
                problems.append(f"[missing {mk}] {qid} -> {p}")
        if TEXT_KEY not in rec or not isinstance(rec[TEXT_KEY], str):
            problems.append(f"[no query] {qid}")
        if BBOX_KEY in rec:
            b = rec[BBOX_KEY]
            if not (isinstance(b, list) and len(b) == 4):
                problems.append(f"[bad bbox] {qid}: {b}")
    return problems


def split_train_val(records: Dict[str, dict], val_ratio: float, seed: int) -> tuple[dict, dict]:
    """按 ID 划分 train/val。"""
    ids = sorted(records.keys())
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_ratio))
    val_ids = set(ids[:n_val])
    train = {k: v for k, v in records.items() if k not in val_ids}
    val = {k: v for k, v in records.items() if k in val_ids}
    return train, val


def visualize(records: Dict[str, dict], data_root: Path, out_dir: Path, n: int = 8) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = list(records.keys())[:n]
    for qid in ids:
        rec = records[qid]
        v_path = data_root / rec[MODALITY_KEYS["visible"]]
        img = cv2.imread(str(v_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = rec.get(BBOX_KEY, [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v * d) for v, d in zip(bbox, (w, h, w, h))]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            rec.get(TEXT_KEY, "")[:60],
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        out_path = out_dir / f"{qid}.jpg"
        cv2.imwrite(str(out_path), img)
    print(f"Saved visualizations to {out_dir}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    print(f"Data root: {data_root}")
    records = load_all(data_root)
    print(f"Total samples: {len(records)}")
    problems = validate(records, data_root)
    if problems:
        print(f"[WARN] 发现 {len(problems)} 个问题（前 10 条）:")
        for p in problems[:10]:
            print("  -", p)
    else:
        print("[OK] 所有样本均通过基础校验。")

    # 划分
    train_json = data_root / "train.json"
    val_json = data_root / "val.json"
    if train_json.exists() and not (data_root / args.src_json).exists():
        print(f"已存在 {train_json.name}，不重复划分。")
    else:
        train, val = split_train_val(records, args.val_ratio, args.seed)
        with open(train_json, "w", encoding="utf-8") as f:
            json.dump(train, f, ensure_ascii=False, indent=2)
        with open(val_json, "w", encoding="utf-8") as f:
            json.dump(val, f, ensure_ascii=False, indent=2)
        print(
            f"Split done: train={len(train)}, val={len(val)} "
            f"-> {train_json.name}, {val_json.name}"
        )

    if not args.no_vis:
        visualize(records, data_root, Path(args.vis_out), n=8)


if __name__ == "__main__":
    main()
