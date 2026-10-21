"""评测脚本（需真实标签）。

赛题官方测试集不公开 bbox 标签；本脚本用于在自带标注的验证集上评测，
输出 ACC@0.5 与平均 IoU，供本地调参。

运行：
    python scripts/evaluate.py --predictions outputs/predictions.json \
        --ground_truth path/to/gt.json
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from src.config import load_config
from src.eval.metrics import acc_at_iou, mean_iou
from src.utils.io import load_json
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="评测 ACC@0.5")
    p.add_argument("--config", default=None)
    p.add_argument("--predictions", required=True, help="预测 JSON 路径")
    p.add_argument("--ground_truth", required=True, help="带 bbox 标签的 JSON 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logging(cfg.logging.level)

    preds = load_json(args.predictions)
    gts = load_json(args.ground_truth)

    # 按 Query ID 对齐，仅统计两文件共有的样本
    common = sorted(set(preds) & set(gts))

    def _extract_box(v):
        if isinstance(v, dict) and "bbox" in v:
            return v["bbox"]
        return v

    pred_list = [_extract_box(preds[k]) for k in common]
    gt_list = [gts[k]["bbox"] for k in common]

    acc = acc_at_iou(pred_list, gt_list, threshold=0.5)
    miou = mean_iou(pred_list, gt_list)

    logger.info("共 %d 条样本参与评测", len(common))
    logger.info("ACC@0.5 = %.4f", acc)
    logger.info("Mean IoU = %.4f", miou)


if __name__ == "__main__":
    main()
