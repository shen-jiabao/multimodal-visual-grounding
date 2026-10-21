from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from src.config import load_config
from src.inference.predictor import run_inference
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="冒烟测试（前 2 张图，供 run_full.sh 前置检查）")
    p.add_argument("--config", default=None, help="配置文件路径（默认 configs/config.yaml）")
    p.add_argument("--data-root", default=None, help="覆盖数据集根目录（绝对路径或相对项目根）")
    p.add_argument("--device", default=None, help="覆盖设备：cpu / cuda / auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.data_root:
        overrides["data.root"] = args.data_root
    if args.device:
        overrides["model.device"] = args.device

    cfg = load_config(args.config, overrides)
    logger = setup_logging(cfg.logging.level)

    predictions = run_inference(cfg, limit_images=2)
    total = len(predictions)
    non_none = sum(1 for v in predictions.values() if v is not None)

    logger.info("冒烟结果：处理 %d 条查询（前 2 图），%d 条得到有效框", total, non_none)
    if total == 0 or non_none == 0:
        logger.error("冒烟失败：无有效预测输出")
        sys.exit(1)
    logger.info("冒烟通过")


if __name__ == "__main__":
    main()
