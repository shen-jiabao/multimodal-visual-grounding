"""端到端 pipeline 验证：训练 + 预测 + 评估。

仅用于流程验证，不追求精度。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from config import (
    CHECKPOINT_DIR,
    DATA_ROOT,
    IMAGE_SIZE,
    MODEL_CFG,
    get_device,
)
from datasets.mmvg_dataset import MMGroundingDataset, collate_fn
from losses.box_loss import GroundingLoss
from models.grounding_model import build_model
from torch.utils.data import DataLoader
from utils.metrics import acc_at_iou


def main():
    print(f"Device: {get_device()}")
    device = torch.device(get_device())

    # 用离线小模型
    cfg = {
        **MODEL_CFG,
        "image_backbone": "resnet10t",
        "image_pretrained": False,
        "text_pretrained": False,
    }
    model = build_model(cfg).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 训练集 = val集 = sample
    train_set = MMGroundingDataset(
        json_path=DATA_ROOT / "train.json",
        data_root=DATA_ROOT, is_train=True, require_bbox=True,
    )
    val_set = MMGroundingDataset(
        json_path=DATA_ROOT / "val.json",
        data_root=DATA_ROOT, is_train=False, require_bbox=True,
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = GroundingLoss()

    print("\n[TRAIN] 5 epochs")
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            rgb = batch["visible"].to(device)
            ir = batch["infrared"].to(device)
            d = batch["depth"].to(device)
            gt = batch["bbox"].to(device)
            out = model(rgb, ir, d, batch["text"])
            pred = out["bbox"]
            loss_dict = loss_fn(pred, gt)
            optim.zero_grad()
            loss_dict["loss"].backward()
            optim.step()
            print(
                f"  ep{epoch} loss={loss_dict['loss'].item():.4f} "
                f"iou={loss_dict['iou'].item():.4f} "
                f"pred={pred[0].tolist()[:4]} gt={gt[0].tolist()[:4]}"
            )

    # 保存 checkpoint
    ckpt_path = CHECKPOINT_DIR / "demo.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": cfg,
            "image_size": IMAGE_SIZE,
        },
        ckpt_path,
    )
    print(f"\nSaved checkpoint to {ckpt_path}")

    # 在 val 上评估
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["visible"].to(device)
            ir = batch["infrared"].to(device)
            d = batch["depth"].to(device)
            gt = batch["bbox"].to(device)
            out = model(rgb, ir, d, batch["text"])
            pred = out["bbox"]
            iou = acc_at_iou(pred.cpu().numpy(), gt.cpu().numpy(), 0.5)
            print(f"[EVAL] ACC@0.5={iou['acc']:.4f}, IoU={iou['miou']:.4f}")

    # 生成预测 JSON
    test_set = MMGroundingDataset(
        json_path=DATA_ROOT / "test.json",
        data_root=DATA_ROOT, is_train=False, require_bbox=False,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    results = {}
    for qid, rec in test_set.records.items():
        results[qid] = {
            "visible": rec.get("visible"),
            "infrared": rec.get("infrared"),
            "depth": rec.get("depth"),
            "query": rec.get("query"),
            "bbox": None,
        }
    with torch.no_grad():
        for batch in test_loader:
            rgb = batch["visible"].to(device)
            ir = batch["infrared"].to(device)
            d = batch["depth"].to(device)
            out = model(rgb, ir, d, batch["text"])
            pred = out["bbox"].cpu().numpy().tolist()
            for qid, b in zip(batch["query_id"], pred):
                results[qid]["bbox"] = [float(x) for x in b]
    out_json = ROOT / "work_dirs" / "demo_result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[PRED] Saved to {out_json}")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
