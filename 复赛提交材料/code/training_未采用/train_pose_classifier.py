# -*- coding: utf-8 -*-
"""训练检测后姿态分类器（正/侧/背，方案 2）。

训练集 JSON 每项需含可见光/红外/深度路径、归一化 bbox 与 pose 标签：
{
  "xxx_001": {
    "visible": "Images/visible/xxx.png",
    "infrared": "Images/infrared/xxx.png",
    "depth": "Images/depth/xxx.png",
    "bbox": [0.10, 0.20, 0.30, 0.50],
    "pose": "front"          // front | side | back
  }
}

说明：
- 训练前一次性抽取全部三模态 patch 到内存，避免每 epoch 重复 IO。
- 按类别分层划分 train/val，每 epoch 对训练集 shuffle，按 batch 前向以稳定梯度。
- CrossEntropyLoss 使用类频率倒数加权，抵消 side 类偏多。

用法：
    python train_pose_classifier.py --train_json pose_train_auto.json \
        --out checkpoints/pose_cls --epochs 20 --batch_size 16
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
import os
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from models.pose_classifier import (
    POSE_CLASSES,
    PoseClassifier,
    _imread_unicode,
    load_ir_gray,
    prepare_patch,
)

# 数据根目录（与主推理一致）
DATA_ROOT = Path(str(ROOT) + r"\初赛数据集-基于大模型的多模态视觉理解与推理")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", type=str, required=True,
                   help="带 pose 标签的训练集 JSON 路径")
    p.add_argument("--out", type=str, default="checkpoints/pose_cls")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=224, help="patch 输入尺寸")
    p.add_argument("--base", type=int, default=32, help="CNN 基础通道数")
    p.add_argument("--batch_size", type=int, default=16, help="训练 batch 大小")
    p.add_argument("--val_ratio", type=float, default=0.15,
                   help="验证集占比（按类别分层）")
    p.add_argument("--seed", type=int, default=42, help="随机种子（切分与洗牌可复现）")
    p.add_argument("--log_interval", type=int, default=10,
                   help="每 N 个 batch 打印一次 loss")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_patch(rec: dict, size: int) -> torch.Tensor | None:
    """读取三模态图并裁剪为一个 patch；读取失败返回 None。"""
    try:
        rgb = _imread_unicode(str(DATA_ROOT / rec["visible"]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ir = load_ir_gray(str(DATA_ROOT / rec["infrared"]))
        depth = _imread_unicode(str(DATA_ROOT / rec["depth"]), cv2.IMREAD_UNCHANGED)
        if depth is None:
            return None
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.uint16)
    except Exception as e:  # noqa: BLE001
        print(f"[LOAD ERR] {rec.get('visible')}: {e}", flush=True)
        return None
    bbox = np.asarray(rec["bbox"], dtype=np.float32).reshape(-1)
    return prepare_patch(rgb, ir, depth, bbox, size=size)


def stratified_split(data: list, val_ratio: float, seed: int):
    """按标签分层划分 train/val。data 元素为 (patch, target)。"""
    by_class = {c: [] for c in range(len(POSE_CLASSES))}
    for patch, target in data:
        by_class[target].append((patch, target))
    rng = random.Random(seed)
    train, val = [], []
    for c in range(len(POSE_CLASSES)):
        items = by_class[c]
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


def evaluate(model, data: list, device, batch_size: int):
    """在给定数据集上评估，返回 (整体准确率, 各类别准召)。"""
    model.eval()
    n_correct = 0
    per_class = {c: {"correct": 0, "total": 0} for c in range(len(POSE_CLASSES))}
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            patches = torch.stack([x[0] for x in batch]).to(device)
            targets = torch.tensor([x[1] for x in batch], device=device)
            logits = model(patches)
            preds = logits.argmax(dim=1)
            n_correct += int((preds == targets).sum().item())
            for t, p in zip(targets.tolist(), preds.tolist()):
                per_class[t]["total"] += 1
                if t == p:
                    per_class[t]["correct"] += 1
    acc = n_correct / max(len(data), 1)
    return acc, per_class


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    with open(args.train_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    # 过滤有效样本
    samples = []
    for qid, rec in records.items():
        pose = rec.get("pose", "").lower().strip()
        if pose not in POSE_CLASSES or "bbox" not in rec:
            continue
        samples.append((qid, rec, POSE_CLASSES.index(pose)))
    print(f"有效样本数: {len(samples)}")
    print(f"类别分布: {dict(Counter(s[2] for s in samples))}")

    # 一次性抽取全部 patch（IO 与裁剪只做一次）
    data = []
    for qid, rec, target in samples:
        patch = load_patch(rec, args.size)
        if patch is None:
            continue
        data.append((patch, target))
    print(f"成功抽取 patch: {len(data)}")

    train_data, val_data = stratified_split(data, args.val_ratio, args.seed)
    print(f"train={len(train_data)} val={len(val_data)}")
    print(f"train 分布: {dict(Counter(t for _, t in train_data))}")
    print(f"val 分布: {dict(Counter(t for _, t in val_data))}")

    model = PoseClassifier(in_chans=5, num_classes=3, base=args.base).to(device)
    # 类别权重：按频率倒数（已过滤返回 torch 对象，需在 device 上）
    counts = Counter(t for _, t in train_data)
    total = len(train_data)
    weights = torch.tensor(
        [total / (len(POSE_CLASSES) * counts.get(c, 1)) for c in range(len(POSE_CLASSES))],
        dtype=torch.float32, device=device,
    )
    print(f"类别权重: {weights.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        # 每 epoch 洗牌，打破类别顺序，避免顺序性灾难遗忘
        random.shuffle(train_data)
        total = 0.0
        n_batch = 0
        t0 = time.time()
        n_steps = (len(train_data) + args.batch_size - 1) // args.batch_size
        for i in range(0, len(train_data), args.batch_size):
            batch = train_data[i:i + args.batch_size]
            patches = torch.stack([x[0] for x in batch]).to(device)
            targets = torch.tensor([x[1] for x in batch], device=device)
            logits = model(patches)
            loss = loss_fn(logits, targets)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total += loss.item()
            n_batch += 1
            if (n_batch % args.log_interval) == 0:
                print(f"[Epoch {epoch}] batch {n_batch}/{n_steps} "
                      f"loss={loss.item():.4f}", flush=True)

        avg_loss = total / max(n_batch, 1)
        # 验证集评估
        val_acc, per_class = evaluate(model, val_data, device, args.batch_size)
        pc_str = " ".join(
            f"{POSE_CLASSES[c]}:{per_class[c]['correct']}/{per_class[c]['total']}"
            for c in range(len(POSE_CLASSES))
        )
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} "
              f"val_acc={val_acc:.4f}  ({pc_str}) "
              f"耗时{time.time() - t0:.1f}s", flush=True)

        ckpt = {
            "state_dict": model.state_dict(),
            "classes": POSE_CLASSES,
            "size": args.size,
            "base": args.base,
            "epoch": epoch,
            "val_acc": val_acc,
        }
        torch.save(ckpt, Path(args.out) / "pose_latest.pth")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(ckpt, Path(args.out) / "pose_best.pth")
            print(f"  -> saved pose_best.pth (val_acc={best_acc:.4f})", flush=True)

    print(f"训练完成。best_val_acc={best_acc:.4f}，权重目录: {args.out}")


if __name__ == "__main__":
    main()