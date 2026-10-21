# -*- coding: utf-8 -*-
"""Grounding DINO 的 LoRA 低秩微调（9.1）训练脚本。

目标：用带 GT bbox 的训练集，冻结基座、只训练 LoRA + 门控参数，
适配三模态输入与空间语义，提升多实例分离与正/侧位姿识别。

用法：
    python train_lora_gd.py \
        --train_json "初赛数据集-.../train.json" \
        --fusion --epochs 3 --lr 1e-4 --out checkpoints/lora_gd

说明：
    - 训练集 JSON 结构与 test.json 相同，但每条额外含 "bbox"（归一化 xyxy）。
    - 逐样本前向，选与 GT IoU 最大的候选框做 L1+IoU+GIoU 回归。
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
import time
from pathlib import Path

import torch
from torchvision.ops import box_iou

from losses.box_loss import GroundingLoss
from models.gd_lora import freeze_except_lora, inject_lora, lora_state_dict
from utils.box_ops import box_cxcywh_to_xyxy

import predict_gd_3modal as P


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", type=str, required=True,
                   help="带 bbox 的训练集 JSON 路径")
    p.add_argument("--out", type=str, default="checkpoints/lora_gd",
                   help="LoRA 权重输出目录")
    p.add_argument("--fusion", action="store_true",
                   help="启用特征层门控融合（9.2）")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--r", type=int, default=8, help="LoRA 秩")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA 缩放")
    p.add_argument("--topk", type=int, default=10,
                   help="每样本用于匹配的候选框数量")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--device", type=str, default=str(P.DEVICE))
    return p.parse_args()


def load_model(args) -> torch.nn.Module:
    """加载 GD 模型，可选特征融合，注入 LoRA 并冻结基座。"""
    model = P.load_model_3modal(feature_fusion=args.fusion)
    inject_lora(model, r=args.r, alpha=args.alpha)
    freeze_except_lora(model)
    model.to(args.device)
    return model


def forward_candidates(model, mm_t, caption, device, topk):
    """前向得到候选框与分数。返回 (topk_xyxy, topk_scores)。"""
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    t = mm_t.unsqueeze(0).to(device)
    if next(model.parameters()).dtype != t.dtype:
        t = t.to(next(model.parameters()).dtype)
    out = model(t, captions=[caption])
    logits = out["pred_logits"][0].sigmoid()   # (N, 256)
    boxes = out["pred_boxes"][0]               # (N, 4) cxcywh
    scores = logits.max(dim=1)[0]              # (N,)
    if scores.numel() == 0:
        return None, None
    topk = min(topk, scores.numel())
    top_idx = scores.topk(topk)[1]
    cand_xyxy = box_cxcywh_to_xyxy(boxes[top_idx])
    return cand_xyxy, scores[top_idx]


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    model = load_model(args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {n_train:,} / {n_total:,} "
          f"({100 * n_train / max(n_total, 1):.2f}%)")

    with open(args.train_json, "r", encoding="utf-8") as f:
        records = json.load(f)
    ids = list(records.keys())
    print(f"训练样本数: {len(ids)}")

    loss_fn = GroundingLoss(lambda_l1=1.0, lambda_iou=2.0, lambda_giou=1.0)
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        n = 0
        t0 = time.time()
        for i, qid in enumerate(ids):
            rec = records[qid]
            if "bbox" not in rec:
                continue
            vis_p = str(P.DATA_ROOT / rec["visible"])
            ir_p = str(P.DATA_ROOT / rec["infrared"])
            dep_p = str(P.DATA_ROOT / rec["depth"])
            try:
                _, mm_t, h, w, _ = P.load_multimodal_image(
                    vis_p, ir_p, dep_p)
            except AssertionError:
                continue

            gt = torch.tensor(rec["bbox"], dtype=torch.float32,
                              device=device).unsqueeze(0)  # (1,4) xyxy
            cand_xyxy, scores = forward_candidates(
                model, mm_t, rec["query"], device, args.topk)
            if cand_xyxy is None or cand_xyxy.shape[0] == 0:
                continue

            # 选与 GT IoU 最大的候选做回归
            ious = box_iou(cand_xyxy, gt).squeeze(1)   # (topk,)
            best = int(ious.argmax())
            pred = cand_xyxy[best].unsqueeze(0)        # (1,4)

            loss = loss_fn(pred, gt)["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optim.step()

            total += loss.item()
            n += 1
            if (i + 1) % args.log_interval == 0:
                el = time.time() - t0
                print(f"[Epoch {epoch}] {i+1}/{len(ids)} "
                      f"loss={loss.item():.4f} ({n / max(el, 1e-6):.1f} it/s)")

        avg = total / max(n, 1)
        print(f"[Epoch {epoch}] avg_loss={avg:.4f}")

        # 保存 latest 与 best
        lora_sd = lora_state_dict(model)
        latest_p = Path(args.out) / "lora_latest.pth"
        torch.save({"lora": lora_sd, "epoch": epoch, "r": args.r,
                    "alpha": args.alpha}, latest_p)
        if avg < best_loss:
            best_loss = avg
            best_p = Path(args.out) / "lora_best.pth"
            torch.save({"lora": lora_sd, "epoch": epoch, "r": args.r,
                        "alpha": args.alpha}, best_p)
            print(f"  -> saved lora_best.pth (loss={best_loss:.4f})")

    print(f"训练完成。best_loss={best_loss:.4f}，权重目录: {args.out}")


if __name__ == "__main__":
    main()