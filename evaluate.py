"""评估：在验证集/测试集（有真值）上计算 ACC@0.5、mIoU。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEFAULT_TEST_JSON,
    DEFAULT_VAL_JSON,
    EVAL_CFG,
    MODEL_CFG,
    get_device,
)
from datasets.mmvg_dataset import MMGroundingDataset, collate_fn
from losses.box_loss import GroundingLoss
from models.grounding_model import build_model
from utils.metrics import aggregate_metrics, acc_at_iou
from train import evaluate as eval_loop  # 复用训练脚本的评估逻辑


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--val_json", type=str, default=DEFAULT_VAL_JSON)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=get_device())
    p.add_argument("--save_json", type=str, default="", help="将预测结果也保存到 JSON")
    p.add_argument(
        "--online", action="store_true",
        help="强制使用预训练 HF 模型（需要联网）。默认离线。",
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    val_set = MMGroundingDataset(
        json_path=Path(args.data_root) / args.val_json,
        data_root=args.data_root,
        is_train=False,
        require_bbox=True,
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("model_cfg") or ck.get("cfg") or MODEL_CFG
    # cfg 可能是 argparse Namespace 转 dict
    if not isinstance(cfg, dict) or "text_max_length" not in cfg:
        cfg = MODEL_CFG
    if not args.online:
        cfg["text_pretrained"] = False
        cfg["image_pretrained"] = False
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    loss_fn = GroundingLoss()
    val_loss, metrics = eval_loop(
        model, loader, loss_fn, device, iou_threshold=EVAL_CFG["iou_threshold"]
    )
    print(f"val_loss={val_loss:.4f}")
    print(
        f"ACC@{EVAL_CFG['iou_threshold']} = {metrics['acc']:.4f} "
        f"({metrics['correct']}/{metrics['num']}), "
        f"mIoU={metrics['miou']:.4f}"
    )

    if args.save_json:
        out = {}
        for batch in loader:
            rgb = batch["visible"].to(device)
            ir = batch["infrared"].to(device)
            d = batch["depth"].to(device)
            out_dict = model(rgb, ir, d, batch["text"])
            pred = out_dict["bbox"].detach().cpu().numpy().tolist()
            for qid, b in zip(batch["query_id"], pred):
                rec = val_set.get_record(qid)
                out[qid] = {
                    "visible": rec.get("visible"),
                    "infrared": rec.get("infrared"),
                    "depth": rec.get("depth"),
                    "query": rec.get("query"),
                    "bbox": [float(x) for x in b],
                }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Predictions saved to {args.save_json}")


if __name__ == "__main__":
    main()
