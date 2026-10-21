"""导出提交包脚本。

将预测结果 JSON 组织成赛题要求的提交格式（保留原始字段，仅填充 bbox），
并压缩为 zip 提交包。

运行：
    python scripts/export_submission.py --predictions outputs/predictions.json
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401

from src.config import load_config
from src.postprocess.bbox import is_valid_bbox, sanitize_bbox
from src.utils.io import load_json, save_json, zip_file
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="导出提交 zip")
    p.add_argument("--config", default=None)
    p.add_argument("--data-root", default=None, help="覆盖数据集根目录（读取原始 queries.json）")
    p.add_argument("--predictions", default=None, help="预测 JSON 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.data_root:
        overrides["data.root"] = args.data_root
    cfg = load_config(args.config, overrides)
    logger = setup_logging(cfg.logging.level)

    pred_path = args.predictions or os.path.join(cfg.output_dir, cfg.output.predictions_json)
    predictions = load_json(pred_path)

    # 原始查询文件（用于还原可见光/红外/深度路径与 query 字段）
    raw = load_json(cfg.queries_path)

    # 无效/缺失预测的中性占位框（默认中心小框 [0.49,0.49,0.51,0.51]），可配置
    fallback_bbox = list(cfg.postprocess.fallback_bbox)

    submission = {}
    invalid = 0
    for qid, item in raw.items():
        bbox = predictions.get(qid)
        if bbox is None or not is_valid_bbox(bbox):
            # 无效/缺失预测：置为配置的中性框（保证评测可解析），并计数
            bbox = fallback_bbox
            invalid += 1
        else:
            bbox = sanitize_bbox(bbox)
        submission[qid] = {
            "visible": item["visible"],
            "infrared": item["infrared"],
            "depth": item["depth"],
            "query": item["query"],
            "bbox": bbox,
        }

    # 写出 JSON 并打包（仅打包 submission.json 单文件，避免递归打包自身）
    out_json = os.path.join(cfg.output_dir, "submission.json")
    save_json(submission, out_json)
    zip_path = os.path.join(cfg.output_dir, cfg.output.submission_zip)
    zip_file(out_json, zip_path, arcname="submission.json")

    logger.info("提交文件已生成：%s（%d 条，其中无效/缺失 %d 条）",
                out_json, len(submission), invalid)
    logger.info("zip 提交包：%s", zip_path)


if __name__ == "__main__":
    main()
