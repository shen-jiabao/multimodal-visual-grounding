# -*- coding: utf-8 -*-
"""半自动姿态标注脚本（生成正/侧/背训练样本）。

思路：
1. 从 query 文本解析朝向词（front/side/back），做「自动」标签来源。
2. 结合检测结果 result_gd_3modal.json 的 bbox，产出 (visible/ir/depth + bbox + pose) 样本。
3. 同一张图、同一个 bbox 出现多种朝向时视为「冲突」，列入人工复核，不自动入库。
4. 生成裁剪图供人工检查；支持 --review_json 对单条做修正/剔除。

用法：
    python annotate_pose.py \
        --test_json "初赛数据集-.../test.json" \
        --result_json result_gd_3modal.json \
        --out pose_train_auto.json \
        --viz_dir _pose_annot_viz
    # 人工修正后再重跑：
    python annotate_pose.py --review_json review.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from models.pose_classifier import (
    POSE_CLASSES,
    _imread_unicode,
    parse_pose_word,
)

DATA_ROOT = Path(r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理")
# 检测无结果时的兜底框，自动跳过
FALLBACK_BBOX = [0.25, 0.25, 0.75, 0.75]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_json", type=str, default=str(DATA_ROOT / "test.json"))
    p.add_argument("--result_json", type=str,
                   default=r"d:\Desktop\2026AiC\result_gd_3modal.json")
    p.add_argument("--out", type=str,
                   default=r"d:\Desktop\2026AiC\pose_train_auto.json")
    p.add_argument("--review_json", type=str, default="",
                   help="人工修正文件：{qid: front|side|back|drop}")
    p.add_argument("--viz_dir", type=str, default="",
                   help="生成裁剪图用于人工检查（为空则跳过）")
    p.add_argument("--ndigits", type=int, default=4,
                   help="bbox 去重时的舍入位数")
    p.add_argument("--pad", type=float, default=0.15,
                   help="可视化裁剪时框外扩比例")
    return p.parse_args()


def _round_box(box, ndigits):
    return tuple(round(float(v), ndigits) for v in box)


def _valid_box(box):
    """判断 bbox 是否有效（4 个值，且不是兜底框）。"""
    if not box or len(box) != 4:
        return False
    if all(abs(float(box[i]) - FALLBACK_BBOX[i]) < 1e-6 for i in range(4)):
        return False
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return False
    return True


def _save_crop(visible_path, bbox, pose, qid, out_dir, pad=0.15):
    """裁剪 RGB 区域并保存，文件名带序号与姿态标签。"""
    img = _imread_unicode(visible_path, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0.0, x1 - pad * bw)
    x2 = min(1.0, x2 + pad * bw)
    y1 = max(0.0, y1 - pad * bh)
    y2 = min(1.0, y2 + pad * bh)
    crop = img[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    if crop.size == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{qid}__{pose}.jpg"
    cv2.imwrite(str(out_path), crop)


def main():
    args = parse_args()

    with open(args.test_json, "r", encoding="utf-8") as f:
        test = json.load(f)
    with open(args.result_json, "r", encoding="utf-8") as f:
        result = json.load(f)

    # 人工修正
    review = {}
    if args.review_json:
        with open(args.review_json, "r", encoding="utf-8") as f:
            review = json.load(f)

    # 1) 自动解析朝向词 + 取 bbox
    candidates = []  # (qid, visible, infrared, depth, bbox, pose_idx)
    auto_count = 0
    for qid, rec in test.items():
        pose_idx = parse_pose_word(rec.get("query", ""))
        if pose_idx is None:
            continue
        auto_count += 1
        # 人工修正优先
        if qid in review:
            rv = review[qid].strip().lower()
            if rv == "drop":
                continue
            if rv in POSE_CLASSES:
                pose_idx = POSE_CLASSES.index(rv)
        box = result.get(qid, {}).get("bbox")
        if not _valid_box(box):
            continue
        candidates.append((qid, rec["visible"], rec["infrared"], rec["depth"],
                           box, pose_idx))

    print(f"query 中含朝向词的条目: {auto_count}")
    print(f"有效候选样本(有合法 bbox): {len(candidates)}")

    # 2) 按 (图片, bbox) 分组检测冲突
    groups = defaultdict(list)
    for qid, vis, ir, dep, box, pose_idx in candidates:
        key = (vis, _round_box(box, args.ndigits))
        groups[key].append((qid, ir, dep, box, pose_idx))

    out_records = []
    conflicts = []
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    idx = 0
    for (vis, box_key), items in groups.items():
        poses = {it[4] for it in items}
        if len(poses) == 1:
            pose_idx = poses.pop()
            # 取该组第一条作为样本（同图同框去重）
            qid, ir, dep, box, _ = items[0]
            rec = {
                "visible": vis,
                "infrared": ir,
                "depth": dep,
                "bbox": list(box_key),
                "pose": POSE_CLASSES[pose_idx],
            }
            out_records.append((qid, rec))
            if viz_dir is not None:
                _save_crop(str(DATA_ROOT / vis), list(box_key),
                           POSE_CLASSES[pose_idx], qid, viz_dir / "ok",
                           pad=args.pad)
        else:
            pose_names = [POSE_CLASSES[p] for p in sorted(poses)]
            conflicts.append({"image": vis, "bbox": list(box_key),
                              "poses": pose_names,
                              "qids": [it[0] for it in items]})
            if viz_dir is not None:
                # 冲突样本另存，便于人工判断
                for qid, ir, dep, box, pi in items:
                    _save_crop(str(DATA_ROOT / vis), list(box_key),
                               POSE_CLASSES[pi], qid, viz_dir / "conflict",
                               pad=args.pad)

    # 3) 写出训练 JSON（键统一用样本序号，保证唯一）
    final = {}
    for qid, rec in out_records:
        key = qid if qid not in final else f"{qid}_dup{idx:03d}"
        final[key] = rec
        idx += 1

    # 复制查询文本信息不进入训练集（训练只需 bbox+pose），保持格式与 train 一致
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    from collections import Counter
    dist = Counter(rec["pose"] for rec in final.values())
    print(f"去重后训练样本数: {len(final)}")
    print(f"姿态分布: {dict(dist)}")
    print(f"冲突(需人工复核): {len(conflicts)}")
    if conflicts:
        c_path = str(Path(args.out).with_suffix(".conflicts.json"))
        with open(c_path, "w", encoding="utf-8") as f:
            json.dump(conflicts, f, ensure_ascii=False, indent=2)
        print(f"冲突明细已写: {c_path}")
        for c in conflicts:
            print(f"  - {c['image']} bbox={c['bbox']} poses={c['poses']} "
                  f"qids={c['qids']}")

    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()