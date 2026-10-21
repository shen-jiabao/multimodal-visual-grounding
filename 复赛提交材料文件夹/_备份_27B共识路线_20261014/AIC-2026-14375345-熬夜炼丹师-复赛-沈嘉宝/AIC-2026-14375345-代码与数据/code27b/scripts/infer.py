"""主推理入口。

运行：
    python scripts/infer.py                 # 全量推理
    python scripts/infer.py --limit 5       # 仅推理前 5 张图（冒烟测试）
    python scripts/infer.py --device cpu    # 强制 CPU

输出：outputs/predictions.json（Query ID -> [x1,y1,x2,y2] 归一化坐标）。
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time

import _bootstrap  # noqa: F401  (注入项目根目录到 sys.path)

from src.config import PROJECT_ROOT, load_config
from src.inference.predictor import run_inference
from src.postprocess.bbox import sanitize_bbox
from src.utils.io import save_json
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="多模态视觉定位推理")
    p.add_argument("--config", default=None, help="配置文件路径（默认 configs/config.yaml）")
    p.add_argument("--limit", type=int, default=None, help="仅处理前 N 张图像（调试用）")
    p.add_argument("--device", default=None, help="覆盖设备：cpu / cuda / auto")
    p.add_argument("--data-root", default=None, help="覆盖数据集根目录（绝对路径或相对项目根）")
    p.add_argument("--output", default=None, help="覆盖输出 JSON 路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {}
    if args.device:
        overrides["model.device"] = args.device
    if args.data_root:
        overrides["data.root"] = args.data_root

    cfg = load_config(args.config, overrides)
    logger = setup_logging(cfg.logging.level)

    # 诊断落盘（纯观测）：cfg.inference.diag_dump=true 时采集模型原始回复/
    # 解析出的 (label,bbox)/匹配失败 label/候选投票，写成 JSONL 供离线归因。
    diag_records: list | None = [] if getattr(cfg.inference, "diag_dump", False) else None

    t0 = time.time()
    out_path = args.output or os.path.join(cfg.output_dir, cfg.output.predictions_json)
    cfg._output_path = out_path  # 供predictor每20张保存
    predictions = run_inference(cfg, limit_images=args.limit, diag=diag_records)

    # 后处理：格式修正，确保所有输出符合赛题规范；
    # 无检测的查询映射为中性占位框（与 export_submission 的缺失处理一致）
    fallback_bbox = list(cfg.postprocess.fallback_bbox)
    predictions = {
        k: sanitize_bbox(v) if v is not None else fallback_bbox
        for k, v in predictions.items()
    }

    out_path = args.output or os.path.join(cfg.output_dir, cfg.output.predictions_json)
    save_json(predictions, out_path)

    # 历史保留：每次运行额外保存一份带时间戳的副本，不覆盖旧结果
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hist_path = os.path.join(cfg.output_dir, "predictions_%s.json" % ts)
        save_json(predictions, hist_path)
        logger.info("历史副本已保留：%s", hist_path)

    # 写诊断 JSONL（每 query 一行；文件很大，不进入提交包）
    if diag_records is not None:
        diag_dir = os.path.join(PROJECT_ROOT, cfg.inference.diag_dir)
        os.makedirs(diag_dir, exist_ok=True)
        tag = os.path.splitext(os.path.basename(out_path))[0]
        diag_path = os.path.join(diag_dir, "diag_raw_%s.jsonl" % tag)
        # 场景名从输出路径里取（如 outputs/<split>/predictions_<desc>.json）
        split = os.path.basename(os.path.dirname(os.path.abspath(out_path))) or "unknown"
        with open(diag_path, "w", encoding="utf-8") as fh:
            for rec in diag_records:
                rec["split"] = split
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("诊断明细已写出：%s（%d 条）", diag_path, len(diag_records))

    logger.info("推理完成：共 %d 条预测，用时 %.1f 秒", len(predictions), time.time() - t0)
    logger.info("结果已保存：%s", out_path)


if __name__ == "__main__":
    main()
