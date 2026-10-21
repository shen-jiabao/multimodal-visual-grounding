"""初赛数据集训练脚本（无 bbox 标注场景）。

策略：
1) 将 queries.json 作为数据源（9555 条查询，来自 2000 张图像）。
2) 对每个样本使用"课程学习"伪标签：
   - 第 0~10 epoch：用文本描述启发式的伪 bbox（中心区域，带随机扰动）。
   - 第 10~30 epoch：用当前模型的预测结果做 EMA 平滑作为伪标签。
   - 第 30 epoch 之后：用当前模型的预测结果作为监督信号。
3) 最终对 queries.json 生成预测 result.json。

用法：
    python train_competition.py
    python train_competition.py --epochs 80 --batch_size 16 --lr 1e-4
    python train_competition.py --ckpt checkpoints/competition/best.pth  # 仅预测
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (
    DATA_ROOT,
    IMAGE_SIZE,
    MODEL_CFG,
    SEED,
    TRAIN_CFG,
    get_device,
)
from models.grounding_model import MultimodalGroundingModel
from losses.box_loss import GroundingLoss


# ============================================================
# 图像加载工具（与 utils/preprocess.py 保持一致）
# ============================================================
def _imread_unicode(path: str, flags: int = 1) -> Optional[np.ndarray]:
    """兼容中文路径的 cv2 图像读取。"""
    import cv2
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, flags)


def load_3modality(data_root: Path, rec: dict, image_size: Tuple[int, int]) -> Optional[Dict[str, np.ndarray]]:
    """加载可见光/红外/深度三模态图像，返回 (H, W, 3) 的 ndarray 字典。"""
    out = {}
    for key in ("visible", "infrared", "depth"):
        rel = rec.get(key, "")
        if not rel:
            return None
        full = str(data_root / rel)
        # 尝试 png/jpg
        img = None
        if os.path.exists(full):
            img = _imread_unicode(full, 1)
        if img is None:
            p = Path(full)
            for ext in (".png", ".jpg", ".jpeg"):
                alt = str(p.with_suffix(ext))
                if os.path.exists(alt):
                    img = _imread_unicode(alt, 1)
                    break
        if img is None:
            return None
        import cv2
        if key == "infrared":
            # 红外：转灰度再复制 3 通道（若已是 3 通道）
            if img.ndim == 3 and img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.stack([gray, gray, gray], axis=-1)
        # 深度保持为 BGR 读取（CV 读取的 16-bit png 会被自动归一化到 8-bit 或保持）
        if key == "depth":
            # 简单处理：若为灰度，复制到 3 通道
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
        out[key] = img
    return out


def ndarr_to_tensor(img: np.ndarray, image_size: Tuple[int, int]) -> torch.Tensor:
    """(H, W, 3) ndarray -> (3, H, W) tensor（归一化到 [0,1] 后再做 ImageNet 标准化）。"""
    import cv2
    img = cv2.resize(img, (image_size[1], image_size[0]))
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t


# ============================================================
# Dataset：从 queries.json 加载，动态生成/读取伪 bbox
# ============================================================
class QueriesDataset(torch.utils.data.Dataset):
    """读取 queries.json，每条样本含 (visible, infrared, depth, query) + 伪 bbox。"""

    def __init__(self, data_root: Path, query_json: str, image_size: Tuple[int, int], pseudo_bbox: Dict[str, List[float]] | None = None):
        self.data_root = data_root
        self.image_size = image_size
        q_path = data_root / query_json
        assert q_path.exists(), f"找不到: {q_path}"
        with open(q_path, "r", encoding="utf-8") as f:
            self.queries: Dict[str, dict] = json.load(f)
        self.ids = sorted(self.queries.keys())
        self.pseudo_bbox = pseudo_bbox or {}
        print(f"[Dataset] 样本数={len(self.ids)} 图像尺寸={image_size}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        qid = self.ids[idx]
        rec = self.queries[qid]
        imgs = load_3modality(self.data_root, rec, self.image_size)
        if imgs is None:
            # 跳过失败样本
            return {"qid": qid, "failed": True}
        vis_t = ndarr_to_tensor(imgs["visible"], self.image_size)
        ir_t = ndarr_to_tensor(imgs["infrared"], self.image_size)
        dep_t = ndarr_to_tensor(imgs["depth"], self.image_size)
        text = rec.get("query", rec.get("text", ""))
        bbox = self.pseudo_bbox.get(qid, [0.25, 0.25, 0.75, 0.75])
        return {
            "qid": qid,
            "visible": vis_t,
            "infrared": ir_t,
            "depth": dep_t,
            "text": text,
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }


def collate_queries(batch: List[Dict]) -> Dict:
    """自定义 collate：过滤失败样本，堆叠张量。"""
    batch = [b for b in batch if not b.get("failed", False)]
    if not batch:
        return {"visible": None}
    out = {
        "qid": [b["qid"] for b in batch],
        "visible": torch.stack([b["visible"] for b in batch]),
        "infrared": torch.stack([b["infrared"] for b in batch]),
        "depth": torch.stack([b["depth"] for b in batch]),
        "text": [b["text"] for b in batch],
        "bbox": torch.stack([b["bbox"] for b in batch]),
    }
    return out


# ============================================================
# 伪标签初始化与更新
# ============================================================
def initialize_pseudo_bbox(queries: Dict[str, dict]) -> Dict[str, List[float]]:
    """对每个样本初始化一个启发式伪框（中心区域 + 随机扰动）。"""
    out = {}
    rng = np.random.RandomState(42)
    for qid in queries.keys():
        # 中心区域：0.35~0.65
        cx = 0.5 + rng.uniform(-0.1, 0.1)
        cy = 0.5 + rng.uniform(-0.1, 0.1)
        w = 0.3 + rng.uniform(-0.05, 0.1)
        h = 0.3 + rng.uniform(-0.05, 0.1)
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(1.0, cx + w / 2)
        y2 = min(1.0, cy + h / 2)
        out[qid] = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
    return out


def update_pseudo_bbox_with_model(
    model: nn.Module,
    dataset: QueriesDataset,
    device: str,
    alpha: float = 0.3,
) -> Dict[str, List[float]]:
    """用当前模型预测所有样本，对伪标签做 EMA 更新。

    new_bbox = alpha * model_pred + (1 - alpha) * old_bbox
    """
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_queries
    )
    new_pseudo = dict(dataset.pseudo_bbox)
    with torch.no_grad():
        for batch in loader:
            if batch["visible"] is None:
                continue
            vis = batch["visible"].to(device)
            ir = batch["infrared"].to(device)
            dep = batch["depth"].to(device)
            out = model(vis, ir, dep, batch["text"])
            pred_bbox = out["bbox"].cpu().numpy()
            for i, qid in enumerate(batch["qid"]):
                old = np.array(new_pseudo.get(qid, [0.25, 0.25, 0.75, 0.75]))
                new = pred_bbox[i]
                merged = alpha * new + (1 - alpha) * old
                new_pseudo[qid] = [round(float(v), 6) for v in merged]
    return new_pseudo


# ============================================================
# 主训练流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    parser.add_argument("--query_json", type=str, default="queries/queries.json")
    parser.add_argument("--epochs", type=int, default=TRAIN_CFG["epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CFG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CFG["lr"])
    parser.add_argument("--weight_decay", type=float, default=TRAIN_CFG["weight_decay"])
    parser.add_argument("--warmup_epochs", type=int, default=TRAIN_CFG["warmup_epochs"])
    parser.add_argument("--warmup_steps", type=int, default=0, help="若>0则覆盖warmup_epochs，直接指定warmup步数")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "checkpoints" / "competition"))
    parser.add_argument("--ckpt", type=str, default="", help="若指定则仅做预测，跳过训练")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE[0], help="输入图像边长（正方形）")
    parser.add_argument("--image_backbone", type=str, default=MODEL_CFG["image_backbone"])
    parser.add_argument("--text_encoder", type=str, default=MODEL_CFG["text_encoder"])
    parser.add_argument("--lambda_l1", type=float, default=TRAIN_CFG["lambda_l1"])
    parser.add_argument("--lambda_iou", type=float, default=TRAIN_CFG["lambda_iou"])
    parser.add_argument("--lambda_giou", type=float, default=TRAIN_CFG["lambda_giou"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # 固定随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(get_device())
    print(f"=== 训练配置 ===", flush=True)
    print(f"  data_root: {args.data_root}", flush=True)
    print(f"  image_backbone: {args.image_backbone}", flush=True)
    print(f"  text_encoder: {args.text_encoder}", flush=True)
    print(f"  epochs: {args.epochs}, batch_size: {args.batch_size}, lr: {args.lr}", flush=True)
    img_size = (args.image_size, args.image_size)
    print(f"  device: {device}, 图像尺寸: {img_size}", flush=True)

    # 加载 queries
    data_root = Path(args.data_root)
    q_path = data_root / args.query_json
    with open(q_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"\n总样本数: {len(queries)}", flush=True)

    # 初始化伪标签
    pseudo_bbox = initialize_pseudo_bbox(queries)
    dataset = QueriesDataset(data_root, args.query_json, img_size, pseudo_bbox)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_queries,
        pin_memory=True,
    )

    # 初始化模型（离线模式：不加载预训练权重）
    model_cfg = dict(MODEL_CFG)
    model_cfg["image_backbone"] = args.image_backbone
    model_cfg["text_encoder"] = args.text_encoder
    model_cfg["text_pretrained"] = False  # 离线
    model_cfg["image_pretrained"] = False  # 离线
    model = MultimodalGroundingModel(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {n_params/1e6:.2f}M", flush=True)

    # 检查点加载
    start_epoch = 0
    if args.ckpt and os.path.exists(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"已加载检查点: {args.ckpt}")
        # 若仅做预测，直接跳到预测
        if args.ckpt and "best.pth" in args.ckpt:
            print("仅预测模式，跳过训练")
            return _predict_and_save(model, dataset, device, args)

    # 优化器 & 调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(loader))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else max(1, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 损失
    loss_fn = GroundingLoss(
        lambda_l1=args.lambda_l1, lambda_iou=args.lambda_iou, lambda_giou=args.lambda_giou
    )

    # ============== 训练循环 ==============
    print("\n=== 开始训练 ===", flush=True)
    t0_total = time.time()
    best_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for step, batch in enumerate(loader):
            if batch["visible"] is None:
                continue
            vis = batch["visible"].to(device, non_blocking=True)
            ir = batch["infrared"].to(device, non_blocking=True)
            dep = batch["depth"].to(device, non_blocking=True)
            gt_bbox = batch["bbox"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(vis, ir, dep, batch["text"])
            loss_dict = loss_fn(out["bbox"], gt_bbox)
            loss = loss_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % args.log_interval == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                print(f"  [Epoch {epoch}/{args.epochs} step {step+1}] "
                      f"loss={loss.item():.4f} (L1={loss_dict['l1'].item():.4f} "
                      f"IoU_loss={loss_dict['iou_loss'].item():.4f} "
                      f"GIoU_loss={loss_dict['giou_loss'].item():.4f}) "
                      f"lr={cur_lr:.2e}", flush=True)

        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f} ({elapsed:.1f}s, {n_batches} batches)", flush=True)

        # 每 10 epoch 更新一次伪标签（课程学习）
        if epoch >= 9 and (epoch + 1) % 10 == 0:
            print(f"  -> 更新伪标签（EMA, alpha={0.3 + min(0.5, epoch/200):.2f}）...", flush=True)
            alpha = 0.3 + min(0.5, epoch / 200)
            new_pseudo = update_pseudo_bbox_with_model(model, dataset, str(device), alpha=alpha)
            dataset.pseudo_bbox = new_pseudo
            n_changes = sum(1 for qid in new_pseudo if qid in pseudo_bbox and new_pseudo[qid] != pseudo_bbox[qid])
            print(f"     更新了 {n_changes} 个样本的伪标签", flush=True)

        # 保存
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "model_cfg": model_cfg,
                    "loss": avg_loss,
                    "image_size": img_size,
                    "pseudo_bbox": dataset.pseudo_bbox,
                },
                Path(args.output_dir) / "best.pth",
            )
            print(f"  -> 保存 best.pth (loss={avg_loss:.4f})", flush=True)

    # 保存最终检查点
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": args.epochs - 1,
            "model_cfg": model_cfg,
            "loss": best_loss,
            "image_size": img_size,
            "pseudo_bbox": dataset.pseudo_bbox,
        },
        Path(args.output_dir) / "latest.pth",
    )
    print(f"\n训练完成！总耗时: {(time.time()-t0_total)/60:.1f} 分钟", flush=True)
    print(f"最佳 loss: {best_loss:.4f}", flush=True)

    # ============== 生成预测文件 ==============
    _predict_and_save(model, dataset, device, args)


def _predict_and_save(model: nn.Module, dataset: QueriesDataset, device: str, args) -> None:
    """对所有样本生成预测并保存为 result.json & result.zip。"""
    print("\n=== 生成预测文件 ===")
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_queries
    )
    results = {}
    with torch.no_grad():
        for batch in loader:
            if batch["visible"] is None:
                continue
            vis = batch["visible"].to(device)
            ir = batch["infrared"].to(device)
            dep = batch["depth"].to(device)
            out = model(vis, ir, dep, batch["text"])
            pred_bbox = out["bbox"].cpu().numpy()
            for i, qid in enumerate(batch["qid"]):
                rec = dataset.queries[qid]
                results[qid] = {
                    "visible": rec.get("visible", ""),
                    "infrared": rec.get("infrared", ""),
                    "depth": rec.get("depth", ""),
                    "query": rec.get("query", ""),
                    "bbox": [round(float(v), 6) for v in pred_bbox[i]],
                }
    # 保存 JSON
    out_json = ROOT / "result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  保存 {len(results)} 条预测 -> {out_json}")

    # 保存 ZIP
    import zipfile
    out_zip = ROOT / "result.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(out_json), arcname="result.json")
    print(f"  打包提交文件 -> {out_zip}")


if __name__ == "__main__":
    main()
