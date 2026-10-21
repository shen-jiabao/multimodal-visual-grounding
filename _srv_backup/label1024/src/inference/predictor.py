"""推理预测模块。

将“数据加载 -> 多模态预处理 -> Grounding DINO 前向 -> 文本与检测框匹配
-> bbox 归一化”串成端到端流程，输出 Query ID -> 归一化 bbox 的预测结果。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from src.config import Config
from src.data.loader import ImageGroup, group_by_image, load_queries, load_sample
from src.data.fusion import build_split_modalities
from src.data.preprocess import build_grounding_input, prepare_depth
from src.inference.matcher import (
    flatten_detections,
    match_by_position,
    match_query_with_depth,
)
from src.models.factory import create_model
from src.postprocess.bbox import normalize_bbox
from src.inference.chains import enforce_ordinal_chains
from src.utils.text import split_by_count

logger = logging.getLogger(__name__)


def run_inference(cfg: Config, limit_images: int | None = None,
                  diag: list | None = None) -> Dict[str, List[float]]:
    """执行全量（或抽样）推理，返回 Query ID -> [x1,y1,x2,y2] 预测。

    流程：双路径加载（load_sample）→ 多模态预处理 → Grounding DINO 前向 →
    flatten_detections 展平 → 三级匹配链（可按 enable_depth_prior 切到深度
    辅助分支）→ bbox 归一化。

    Args:
        diag: 传入 list 时，额外把每图每 query 的诊断记录 append 进去
            （需 ``cfg.inference.diag_dump=true`` 才会采集模型侧明细）。
            纯观测：不影响返回值与 predictions.json 的结构。
    """
    samples = load_queries(cfg)
    groups = group_by_image(samples)
    image_keys = list(groups.keys())
    if limit_images is not None:
        image_keys = image_keys[:limit_images]
    logger.info("共 %d 张图像，%d 条查询；本次处理 %d 张图像",
                len(groups), len(samples), len(image_keys))

    model = create_model(cfg)
    predictions: Dict[str, List[float]] = {}
    collect = diag is not None and bool(getattr(cfg.inference, "diag_dump", False))

    for idx, key in enumerate(image_keys):
        grp = groups[key]
        # 双路径加载（含 depth_is_metric 标记；JPG 深度取 R 通道）
        sample = load_sample(grp.samples[0])
        rgb = sample["rgb"]

        depth_info = prepare_depth(sample)
        if cfg.inference.multimodal_input_mode == "split" and depth_info.get("norm_lin") is not None:
            input_rgb = rgb
            extra_images = build_split_modalities(
                rgb, sample["infrared"],
                depth_info["norm_lin"], depth_info["valid_mask"],
            )
        else:
            extra_images = None
            input_rgb = rgb

        # 同图查询合并前向：先按 max_queries_per_prompt 条数分块（消费该配置）。
        texts = [s.query for s in grp.samples]
        detections: List[dict] = []
        model_diag: List[dict] = []
        # 记录每条 query 落在哪个分块的哪一次模型调用（多图共享一次调用的情况）
        chunk_of_query: Dict[str, int] = {}
        for ci, chunk in enumerate(split_by_count(texts,
                                                  cfg.inference.max_queries_per_prompt)):
            for q in chunk:
                chunk_of_query.setdefault(q, ci)
            detections += flatten_detections(
                model.predict_boxes(
                    input_rgb, chunk,
                    box_threshold=cfg.model.box_threshold,
                    text_threshold=cfg.model.text_threshold,
                    extra_images=extra_images,
                )
            )
            if collect:
                d = getattr(model, "_diag", None)
                if d:
                    model_diag.append({
                        "chunk_index": ci, "chunk_size": len(chunk),
                        "prompt": d.get("prompt"), "images": d.get("images"),
                        "replies": d.get("replies", []),
                        "reply_lens": [len(r or "") for r in d.get("replies", [])],
                        # 回复尾部截断（最后 200 字符）：判断"JSON 是否被截断"的唯一证据
                        "reply_heads": [(r or "")[-200:] for r in d.get("replies", [])],
                        # 未闭合的 "{" 数量：>0 即 JSON 不完整（截断的直接证据）
                        "brace_unclosed": [(r or "").count("{") - (r or "").count("}")
                                           for r in d.get("replies", [])],
                        "pairs": d.get("pairs", []),
                        "unmatched": d.get("unmatched", []),
                        "matched": d.get("matched", 0),
                        "dropped_invalid": d.get("dropped_invalid", 0),
                        "candidates": d.get("candidates", {}),
                        "votes": d.get("votes", {}),
                        # 模式③ ID 化的两个归因字段（多变量同轮上云时唯一凭证）：
                        # id_base 由整条回复的 ID 全集反推（0/1），propagated 是同
                        # 文本 query 弱占优补齐的条数。没有它们，事后无法区分
                        # "ID 化把占位框修好了" 与 "ID 编号漂移反而丢框"。
                        "id_base": d.get("id_base", 1),
                        "propagated": d.get("propagated", 0),
                    })

        h, w = rgb.shape[:2]
        for s in grp.samples:
            det_pos = None
            if cfg.inference.enable_depth_prior:
                det = match_query_with_depth(s.query, detections, depth_info)
                if collect:
                    # 反事实留证（纯观测）：同一次调用下「仅位置匹配」会给出哪个框。
                    # 不参与 det / predictions，只为让深度先验可离线分解——多变量同轮
                    # 上云时，用它在本地重建「关掉深度先验」的那一版预测。
                    det_pos = match_by_position(s.query, detections)
            else:
                det = match_by_position(s.query, detections)

            if det is None and cfg.inference.fallback == "top1" and detections:
                det = max(detections, key=lambda d: d["score"])

            if det is None:
                predictions[s.query_id] = None
            else:
                predictions[s.query_id] = normalize_bbox(
                    det["bbox_pixel"], w, h
                ).tolist()

            if collect:
                ci = chunk_of_query.get(s.query, 0)
                md = model_diag[ci] if ci < len(model_diag) else {}
                diag.append({
                    "query_id": s.query_id,
                    "visible": s.visible_path,
                    "query": s.query,
                    "chunk_index": ci,
                    "chunk_size": md.get("chunk_size"),
                    "matched": det is not None,
                    "pre_chain_box": None if det is None
                    else [float(v) for v in predictions[s.query_id]],
                    # 反事实留证：同上一次模型调用下「只用位置匹配」的框（unit 坐标）。
                    # enable_depth_prior=false 时恒为 None（两条路径同源）。
                    "pos_box": None if det_pos is None else normalize_bbox(
                        det_pos["bbox_pixel"], w, h).tolist(),
                    "enable_depth_prior": bool(cfg.inference.enable_depth_prior),
                    "reply_lens": md.get("reply_lens", []),
                    "reply_heads": md.get("reply_heads", []),
                    "brace_unclosed": md.get("brace_unclosed", []),
                    "unmatched_labels": md.get("unmatched", []),
                    "parsed_pairs": md.get("pairs", []),
                    "votes": (md.get("votes") or {}).get(s.query, None),
                })

        # 序数链修正（跨查询一致性约束，零样本）：默认 off，不改变现有结果
        if cfg.inference.chain_enforce in ("sort", "project"):
            qids = [s.query_id for s in grp.samples]
            fixed = enforce_ordinal_chains(
                [s.query for s in grp.samples],
                [predictions[q] for q in qids],
                mode=cfg.inference.chain_enforce,
                gap_ratio=cfg.inference.chain_gap_ratio,
                max_shift=(cfg.inference.chain_max_shift or None),
            )
            for q, b in zip(qids, fixed):
                if b is not None:
                    predictions[q] = [float(v) for v in b]

        if collect:
            # 补齐链修正后的最终框（供离线量测修正本身的影响）
            for rec in diag[-len(grp.samples):]:
                rec["final_box"] = (None if predictions.get(rec["query_id"]) is None
                                    else [float(v) for v in predictions[rec["query_id"]]])

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_keys):
            logger.info("推理进度 %d/%d", idx + 1, len(image_keys))
            # 每20张保存一次，防止断网丢失
            try:
                import json as _json
                from src.utils.io import save_json as _save
                _out = getattr(cfg, "_output_path", None)
                if _out:
                    _save(predictions, _out)
            except Exception:
                pass

    return predictions
