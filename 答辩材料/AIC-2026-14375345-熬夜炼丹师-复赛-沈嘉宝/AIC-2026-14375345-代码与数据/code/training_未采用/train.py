"""训练入口。

用法：
    python train.py [--epochs N] [--batch_size N] [--lr 1e-4] ...

可选项：
    --data_root  数据根目录（默认 config.DATA_ROOT）
    --train_json 训练集 JSON 路径（默认 train.json）
    --val_json   验证集 JSON 路径（默认 val.json）
    --resume     恢复训练的 checkpoint 路径
    --output_dir 输出目录
    --device     cuda / cpu
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
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEFAULT_TRAIN_JSON,
    DEFAULT_VAL_JSON,
    IMAGE_SIZE,
    MODEL_CFG,
    SEED,
    TRAIN_CFG,
    get_device,
)
from datasets.mmvg_dataset import MMGroundingDataset, collate_fn
from losses.box_loss import GroundingLoss
from models.grounding_model import build_model
from utils.metrics import aggregate_metrics, acc_at_iou


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(args) -> tuple[DataLoader, DataLoader | None]:
    train_set = MMGroundingDataset(
        json_path=Path(args.data_root) / args.train_json,
        data_root=args.data_root,
        is_train=True,
        require_bbox=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = None
    val_json = Path(args.data_root) / args.val_json
    if val_json.exists():
        val_set = MMGroundingDataset(
            json_path=val_json,
            data_root=args.data_root,
            is_train=False,
            require_bbox=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    return train_loader, val_loader


def train_one_epoch(model, loader, optim, scheduler, loss_fn, device, epoch, cfg, log_interval):
    model.train()
    total_loss = 0.0
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        rgb = batch["visible"].to(device, non_blocking=True)
        ir = batch["infrared"].to(device, non_blocking=True)
        d = batch["depth"].to(device, non_blocking=True)
        bbox_gt = batch["bbox"].to(device, non_blocking=True)
        out = model(rgb, ir, d, batch["text"])
        pred = out["bbox"]
        loss_dict = loss_fn(pred, bbox_gt)
        loss = loss_dict["loss"]
        optim.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optim.step()
        total_loss += loss.item() * rgb.size(0)
        n += rgb.size(0)
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"[Epoch {epoch}] step {step+1}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"iou={loss_dict['iou'].item():.4f} "
                f"({n / max(elapsed,1e-6):.1f} it/s)"
            )
    if scheduler is not None:
        scheduler.step()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, iou_threshold=0.5):
    model.eval()
    total_loss = 0.0
    n = 0
    metric_records = []
    for batch in loader:
        rgb = batch["visible"].to(device, non_blocking=True)
        ir = batch["infrared"].to(device, non_blocking=True)
        d = batch["depth"].to(device, non_blocking=True)
        bbox_gt = batch["bbox"].to(device, non_blocking=True)
        out = model(rgb, ir, d, batch["text"])
        pred = out["bbox"]
        loss_dict = loss_fn(pred, bbox_gt)
        total_loss += loss_dict["loss"].item() * rgb.size(0)
        n += rgb.size(0)
        pred_np = pred.detach().cpu().numpy()
        gt_np = bbox_gt.detach().cpu().numpy()
        metric_records.append(acc_at_iou(pred_np, gt_np, iou_threshold=iou_threshold))
    avg_loss = total_loss / max(n, 1)
    metrics = aggregate_metrics(metric_records)
    return avg_loss, metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--train_json", type=str, default=DEFAULT_TRAIN_JSON)
    p.add_argument("--val_json", type=str, default=DEFAULT_VAL_JSON)
    p.add_argument("--output_dir", type=str, default=str(CHECKPOINT_DIR / "exp"))
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CFG["batch_size"])
    p.add_argument("--num_workers", type=int, default=TRAIN_CFG["num_workers"])
    p.add_argument("--lr", type=float, default=TRAIN_CFG["lr"])
    p.add_argument("--weight_decay", type=float, default=TRAIN_CFG["weight_decay"])
    p.add_argument("--warmup_epochs", type=int, default=TRAIN_CFG["warmup_epochs"])
    p.add_argument("--grad_clip", type=float, default=TRAIN_CFG["grad_clip"])
    p.add_argument("--log_interval", type=int, default=TRAIN_CFG["log_interval"])
    p.add_argument("--lambda_l1", type=float, default=TRAIN_CFG["lambda_l1"])
    p.add_argument("--lambda_iou", type=float, default=TRAIN_CFG["lambda_iou"])
    p.add_argument("--lambda_giou", type=float, default=TRAIN_CFG["lambda_giou"])
    p.add_argument("--device", type=str, default=get_device())
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--offline", action="store_true",
        help="不下载预训练权重（HF/timm）。用小模型或随机初始化训练。",
    )
    p.add_argument(
        "--image_backbone", type=str, default=MODEL_CFG["image_backbone"],
        help="timm 中的 backbone 名称（默认 resnet18）",
    )
    p.add_argument(
        "--text_encoder", type=str, default=MODEL_CFG["text_encoder"],
        help="HuggingFace 中的文本编码器（默认 distilbert-base-uncased）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader = build_dataloaders(args)
    print(
        f"Train: {len(train_loader.dataset)} samples, "
        f"Val: {0 if val_loader is None else len(val_loader.dataset)} samples"
    )

    model_cfg = {**MODEL_CFG}
    if args.offline:
        model_cfg["text_pretrained"] = False
        model_cfg["image_pretrained"] = False
    if args.image_backbone:
        model_cfg["image_backbone"] = args.image_backbone
    if args.text_encoder:
        model_cfg["text_encoder"] = args.text_encoder

    model = build_model(model_cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"  text_pretrained={model_cfg['text_pretrained']}, "
          f"image_pretrained={model_cfg['image_pretrained']}, "
          f"backbone={model_cfg['image_backbone']}, "
          f"text_encoder={model_cfg['text_encoder']}")

    loss_fn = GroundingLoss(
        lambda_l1=args.lambda_l1,
        lambda_iou=args.lambda_iou,
        lambda_giou=args.lambda_giou,
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    # 线性 warmup + cosine
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)
    total_steps = max(1, args.epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)

    start_epoch = 0
    best_acc = -1.0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        optim.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["scheduler"])
        start_epoch = ck.get("epoch", 0) + 1
        best_acc = ck.get("best_acc", -1.0)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}")

    log_path = Path(args.output_dir) / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write(f"\n==== start {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
    log_f.flush()

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optim, scheduler, loss_fn, device, epoch,
            vars(args), args.log_interval,
        )
        msg = f"[Epoch {epoch}] train_loss={train_loss:.4f}"
        if val_loader is not None:
            val_loss, val_metrics = evaluate(
                model, val_loader, loss_fn, device, iou_threshold=0.5
            )
            msg += f" | val_loss={val_loss:.4f} val_acc={val_metrics['acc']:.4f} val_miou={val_metrics['miou']:.4f}"
            log_f.write(msg + "\n")
            log_f.flush()
            print(msg)
            cur_acc = val_metrics["acc"]
        else:
            log_f.write(msg + "\n")
            log_f.flush()
            print(msg)
            cur_acc = -train_loss  # 训练损失越小越好

        # 保存 latest
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
                "cfg": vars(args),
                "model_cfg": model_cfg,
                "image_size": IMAGE_SIZE,
            },
            Path(args.output_dir) / "latest.pth",
        )
        # 保存最优
        if cur_acc > best_acc:
            best_acc = cur_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "cfg": vars(args),
                    "model_cfg": model_cfg,
                    "image_size": IMAGE_SIZE,
                },
                Path(args.output_dir) / "best.pth",
            )
            print(f"  -> saved best.pth (acc={best_acc:.4f})")
    log_f.close()
    print(f"Training done. Best acc = {best_acc:.4f}")


if __name__ == "__main__":
    main()
