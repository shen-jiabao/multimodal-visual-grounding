"""推理入口：读取测试集 JSON，预测所有样本 bbox，输出官方要求的 JSON。

用法：
    python predict.py --ckpt checkpoints/exp/best.pth --test_json test.json --output result.json

输出 JSON 格式：
{
  "QueryID": {
    "visible": "...",
    "infrared": "...",
    "depth": "...",
    "query": "...",
    "bbox": [x1, y1, x2, y2]   # 归一化 [0, 1]
  },
  ...
}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    DEFAULT_TEST_JSON,
    MODEL_CFG,
    get_device,
)
from datasets.mmvg_dataset import MMGroundingDataset, collate_fn
from models.grounding_model import build_model
from utils.box_ops import sanitize_box


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    p.add_argument("--test_json", type=str, default=DEFAULT_TEST_JSON)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default=get_device())
    p.add_argument(
        "--output",
        type=str,
        default="result.json",
        help="输出 JSON 路径。",
    )
    p.add_argument(
        "--save_zip", action="store_true", help="是否同时保存为 zip 压缩包"
    )
    p.add_argument(
        "--online", action="store_true",
        help="强制使用预训练 HF 模型（需要联网）。默认离线。",
    )
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    test_set = MMGroundingDataset(
        json_path=Path(args.data_root) / args.test_json,
        data_root=args.data_root,
        is_train=False,
        require_bbox=False,
    )
    loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 先读 checkpoint 获取 cfg
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("model_cfg") or ck.get("cfg") or MODEL_CFG
    if not isinstance(cfg, dict) or "text_max_length" not in cfg:
        cfg = MODEL_CFG
    if not args.online:
        cfg["text_pretrained"] = False
        cfg["image_pretrained"] = False
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    results: Dict[str, dict] = {}
    # 保留原始 JSON 字段（除 bbox 外）
    for qid, rec in test_set.records.items():
        results[qid] = {
            "visible": rec.get("visible"),
            "infrared": rec.get("infrared"),
            "depth": rec.get("depth"),
            "query": rec.get("query"),
            "bbox": None,  # 待填
        }

    for batch in loader:
        rgb = batch["visible"].to(device)
        ir = batch["infrared"].to(device)
        d = batch["depth"].to(device)
        out = model(rgb, ir, d, batch["text"])
        pred = out["bbox"].detach().cpu()
        # 修正 + clamp
        pred = sanitize_box(pred)
        pred_np = pred.numpy().tolist()
        for qid, b in zip(batch["query_id"], pred_np):
            results[qid]["bbox"] = [float(x) for x in b]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions to {out_path} ({len(results)} samples)")

    if args.save_zip:
        import zipfile
        zip_path = out_path.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(out_path, arcname=out_path.name)
        print(f"Zip saved to {zip_path}")


if __name__ == "__main__":
    main()
