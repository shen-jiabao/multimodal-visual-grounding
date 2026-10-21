# -*- coding: utf-8 -*-
"""通用检测+可视化：对指定 test.json 样本运行完整候选管线并可视化框位置。

用法：
    python detect_viz.py --qid 002004_001 002004_002 [--fusion] [-o viz_out]

对每个 qid 输出两张图：
    {qid}_init.png  初始检测框（合并前后对比基准）
    {qid}_pool.png  候选扩充/过滤后的候选框（绿=候选，红=最终选中）

依赖 viz_boxes.draw_boxes 作为通用绘图，避免重复实现。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import predict_gd_3modal as P
from viz_boxes import draw_boxes

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def imread_unicode(path: str) -> np.ndarray:
    """兼容中文路径读取图片（BGR）。"""
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def save_png(img: np.ndarray, path: str) -> None:
    """兼容中文路径保存 PNG。"""
    cv2.imencode(".png", img)[1].tofile(path)


def run_pipeline(model, rec):
    """对单条样本执行完整推理管线，返回候选框、分数、最终框及中间量。

    Returns:
        dict: init_boxes / pool_boxes / pool_scores / final_box / h / w
    """
    query = rec["query"]
    vis_p = str(P.DATA_ROOT / rec["visible"])
    ir_p = str(P.DATA_ROOT / rec["infrared"])
    dep_p = str(P.DATA_ROOT / rec["depth"])

    image_source, mm_t, h, w, depth_full = P.load_multimodal_image(
        vis_p, ir_p, dep_p)
    boxes, logits = P.predict_multimodal(model, mm_t, query)
    boxes_norm = (P.to_xyxy_norm(boxes, h, w)
                  if len(boxes) > 0 else np.empty((0, 4)))
    init_boxes = boxes_norm.copy()

    # 空间 query：类别名词重检测扩充候选
    if P.parse_spatial_criterion(query) is not None:
        cls = P.extract_category(query)
        if cls.lower() != query.lower().strip().rstrip("."):
            b2, l2 = P.predict_multimodal(model, mm_t, cls)
            if len(b2) > 0:
                n2 = P.to_xyxy_norm(b2, h, w)
                if len(boxes_norm) > 0:
                    boxes_norm = np.concatenate([boxes_norm, n2], axis=0)
                    logits = torch.cat([logits, l2])
                else:
                    boxes_norm, logits = n2, l2

    # 候选扩充/切分/过滤（与主流程一致）
    boxes_norm, logits = P.expand_candidates(
        model, mm_t, boxes_norm, logits, query, h, w)
    boxes_norm, logits = P.split_merged_by_depth(
        boxes_norm, logits, depth_full, h, w)
    boxes_norm, logits = P.filter_tall_boxes(
        boxes_norm, logits, query, h, w)

    sel = None
    if len(boxes_norm) > 0:
        # 颜色+空间可组合选择（与主流程一致）
        sel = P.select_color_then_spatial(
            boxes_norm, logits, image_source, depth_full, query, h, w)
    final_box = P.sanitize_box(sel).tolist() if sel is not None else None

    # 颜色属性匹配评分（供观察/验证）
    color = P.parse_target_color(query)
    color_scores = None
    if color is not None and len(boxes_norm) > 0:
        color_scores = np.array([P.box_color_score(b, image_source, color)
                                 for b in boxes_norm])

    return {
        "image": imread_unicode(vis_p),
        "init_boxes": init_boxes,
        "pool_boxes": boxes_norm,
        "pool_scores": logits.cpu().detach().numpy(),
        "final_box": final_box,
        "color": color,
        "color_scores": color_scores,
        "h": h,
        "w": w,
        "query": query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qid", nargs="+", required=True,
                        help="待检测的样本 id（test.json 的 key）")
    parser.add_argument("--fusion", action="store_true",
                        help="用特征层门控融合加载模型")
    parser.add_argument("-o", "--out_dir", default="viz_out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("加载 Grounding DINO ...", flush=True)
    model = P.load_model_3modal(feature_fusion=args.fusion)
    model.eval()
    print(f"模型就绪，设备 {P.DEVICE}", flush=True)

    test = json.load(open(str(P.DATA_ROOT / "test.json"), encoding="utf-8"))

    for qid in args.qid:
        if qid not in test:
            print(f"[跳过] {qid} 不存在")
            continue
        rec = test[qid]
        out = run_pipeline(model, rec)
        query, h, w = out["query"], out["h"], out["w"]

        print("=" * 72)
        print(f"[{qid}] {query}")
        print(f"  初始检测 {len(out['init_boxes'])} 框:")
        for b in out["init_boxes"]:
            print(f"    {[round(float(v), 3) for v in b]}  "
                  f"高宽比={P._box_pixel_aspect(b, h, w):.2f}")
        print(f"  颜色属性: {out['color']}")
        print(f"  候选池 {len(out['pool_boxes'])} 框:")
        for i, b in enumerate(out["pool_boxes"]):
            cs = (f"  color={float(out['color_scores'][i]):.3f}"
                  if out["color_scores"] is not None else "")
            print(f"    [{i}] {[round(float(v), 3) for v in b]}  "
                  f"score={float(out['pool_scores'][i]):.3f}{cs}")
        print(f"  最终选框: {out['final_box']}")

        # 初始框（基准）
        save_png(draw_boxes(out["image"], out["init_boxes"]),
                 str(out_dir / f"{qid}_init.png"))
        # 候选池 + 最终高亮
        labels = [f"[{i}] {float(out['pool_scores'][i]):.2f}"
                  for i in range(len(out["pool_boxes"]))]
        pool_img = draw_boxes(out["image"], out["pool_boxes"],
                              labels=labels, final_box=out["final_box"])
        save_png(pool_img, str(out_dir / f"{qid}_pool.png"))
        saved = [f"{qid}_init.png", f"{qid}_pool.png"]
        # 仅最终选中框（无候选干扰），供直接观察
        if out["final_box"] is not None:
            final_img = draw_boxes(out["image"], [], final_box=out["final_box"])
            save_png(final_img, str(out_dir / f"{qid}_final.png"))
            saved.append(f"{qid}_final.png")
        print(f"  已保存: {', '.join(str(out_dir / s) for s in saved)}")

    print("done")


if __name__ == "__main__":
    main()