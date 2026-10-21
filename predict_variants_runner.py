# -*- coding: utf-8 -*-
"""Grounding DINO / LocateAnything 变体验证 runner 骨架。

复用 predict_gd_3modal 的检测候选流水线，但把「检测 caption」与「选择约束 caption」
解耦，供结构化变体轨（core/spatial/ordinal）做 30 条对比验证。

复用接口（准确签名来自 predict_gd_3modal.py）：
    load_model_3modal(feature_fusion=False)                      -> model
    load_multimodal_image(rgb_path, ir_path, depth_path)         -> (image_source, mm_t, h, w, depth_full)
    predict_multimodal(model, image, caption)                    -> (boxes_cxywh, logits)
    to_xyxy_norm(boxes_cxywh, h, w)                              -> ndarray (N,4) [x1,y1,x2,y2]
    parse_spatial_criterion(caption)                             -> (key, order, rank) | None
    extract_category(caption)                                    -> str
    expand_candidates(model, mm_t, boxes_norm, logits, query, h, w)   -> (boxes_norm, logits)
    split_merged_by_depth(boxes_norm, logits, depth_full, h, w)  -> (boxes_norm, logits)
    filter_tall_boxes(boxes_norm, logits, query, h, w)           -> (boxes_norm, logits)
    select_color_then_spatial(boxes_xyxy, logits, image_rgb, depth_raw, caption, h, w) -> (4,) | None
    sanitize_box(b)                                              -> ndarray

用法：
    python predict_variants_runner.py --dry          # 只打印 30 条变体轨映射，不加载模型
    python predict_variants_runner.py                # 实际跑 GD 结构化轨对比（需 GPU）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import predict_gd_3modal as P  # 复用入口模块（import 仅注册符号，不建模型）

DATA_ROOT = Path("d:/Desktop/2026AiC/初赛数据集-基于大模型的多模态视觉理解与推理")
TEST_JSON = DATA_ROOT / "test.json"
VARIANTS_JSON = Path("d:/Desktop/2026AiC/variants_30.json")

FALLBACK_BBOX = [0.25, 0.25, 0.75, 0.75]


# ----------------------------- 分支判定 -----------------------------
def describe_branch(meta: dict) -> str:
    """根据槽位 meta 判断该样本走哪个处理分支。"""
    if meta.get("is_large_region"):
        return "large_region(none-gd)"
    if meta.get("count") == "all":
        return "collection_all(none-gd)"
    if meta.get("ordinal") is not None:
        return "ordinal(gd-spatial)"
    if meta.get("relation"):
        return "relation(locate-spatial)"
    return "plain"


def build_tracks(variant: dict) -> dict:
    """返回该样本的对照轨与结构化轨 spec。

    origin / struct 各含：
      detect  : 喂给预测/候选扩充的 caption
      select  : 喂给颜色/空间选择规则的 caption
      branch  : 分支说明
    """
    meta = variant["meta"]
    orig = variant["original"]
    branch = describe_branch(meta)
    # 结构化轨：检测用变体，选择约束始终用原句（保留序数/颜色/空间语义）
    struct_detect = variant["core"]
    if branch.startswith("relation"):
        struct_detect = variant["spatial"]  # 实体关系交 Locate 的 spatial 变体
    return {
        "origin":  {"detect": orig, "select": orig, "branch": "control"},
        "struct":  {"detect": struct_detect, "select": orig, "branch": branch},
    }


# ----------------------------- 检测/选择解耦的推理 -----------------------------
def process_one(model, rec, vis_p, ir_p, dep_p, detect, select):
    """复刻 predict_gd_3modal.main 的候选流水线，但检测/选择 caption 分离。"""
    image_source, mm_t, h, w, depth_full = P.load_multimodal_image(vis_p, ir_p, dep_p)
    boxes, logits = P.predict_multimodal(model, mm_t, detect)
    boxes_norm = (P.to_xyxy_norm(boxes, h, w)
                  if len(boxes) > 0 else np.empty((0, 4)))

    # 空间句：用类别名词重检测扩充候选实例池（与主循环一致）
    if P.parse_spatial_criterion(detect) is not None:
        cls = P.extract_category(detect)
        if cls.lower() != detect.lower().strip().rstrip("."):
            b2, l2 = P.predict_multimodal(model, mm_t, cls)
            if len(b2) > 0:
                n2 = P.to_xyxy_norm(b2, h, w)
                if len(boxes_norm) > 0:
                    boxes_norm = np.concatenate([boxes_norm, n2], axis=0)
                    logits = torch.cat([logits, l2])
                else:
                    boxes_norm, logits = n2, l2

    boxes_norm, logits = P.expand_candidates(model, mm_t, boxes_norm, logits,
                                             detect, h, w)
    boxes_norm, logits = P.split_merged_by_depth(boxes_norm, logits,
                                                 depth_full, h, w)
    boxes_norm, logits = P.filter_tall_boxes(boxes_norm, logits, detect, h, w)

    if len(boxes_norm) == 0:
        return {"bbox": FALLBACK_BBOX, "n_cand": 0}
    sel = P.select_color_then_spatial(boxes_norm, logits, image_source,
                                      depth_full, select, h, w)
    if sel is None:
        return {"bbox": FALLBACK_BBOX, "n_cand": int(len(boxes_norm))}
    return {"bbox": P.sanitize_box(sel).tolist(),
            "n_cand": int(len(boxes_norm))}


# ----------------------------- 主流程 -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true",
                        help="只打印 30 条变体轨映射，不加载模型/不推理")
    parser.add_argument("--test_json", default=str(TEST_JSON))
    parser.add_argument("--variants", default=str(VARIANTS_JSON))
    parser.add_argument("--out", default="variants_gd_30.json")
    args = parser.parse_args()

    variants = json.load(open(args.variants, encoding="utf-8"))
    test_data = json.load(open(args.test_json, encoding="utf-8"))

    # 接口自检：确保所有复用符号存在
    required = ["load_multimodal_image", "predict_multimodal", "to_xyxy_norm",
                "parse_spatial_criterion", "extract_category",
                "expand_candidates", "split_merged_by_depth",
                "filter_tall_boxes", "select_color_then_spatial", "sanitize_box"]
    missing = [n for n in required if not hasattr(P, n)]
    if missing:
        print("[FATAL] predict_gd_3modal 缺失符号:", missing, file=sys.stderr)
        return 2

    # 打印 30 条轨映射
    plan = {}
    for qid, v in variants.items():
        rec = test_data.get(qid)
        if rec is None:
            print(f"[WARN] {qid} 不在 test.json，跳过")
            continue
        plan[qid] = build_tracks(v)

    print(f"共 {len(plan)} 条（GD 结构化轨 spec）")
    print("=" * 78)
    for qid, trk in plan.items():
        ov = trk["origin"]["detect"]
        sv = trk["struct"]["detect"]
        print(f"--- {qid}  [{trk['struct']['branch']}]")
        print(f"    select(原句)        : {trk['origin']['select'][:70]}")
        print(f"    struct.detect={sv!r}")
        if trk["struct"]["detect"] != trk["origin"]["detect"]:
            print(f"    origin.detect={ov[:70]!r}")

    if args.dry:
        print("\n[DRY] 未加载模型、未推理，仅输出 spec。指令 'python' 重跑即真实推理（需 GPU）。")
        return 0

    # 真实推理：加载模型并在 30 条上跑对照 vs 结构化
    torch.manual_seed(0)
    np.random.seed(0)
    print("\n加载 Grounding DINO（5 通道早期融合）...")
    model = P.load_model_3modal(feature_fusion=False)

    out = {}
    for qid, trk in plan.items():
        rec = test_data[qid]
        vis = str(DATA_ROOT / rec["visible"])
        ir = str(DATA_ROOT / rec["infrared"])
        dep = str(DATA_ROOT / rec["depth"])
        r = {}
        for name, spec in trk.items():
            r[name] = process_one(model, rec, vis, ir, dep,
                                  spec["detect"], spec["select"])
        out[qid] = r
        print(f"[{qid}] branch={trk['struct']['branch']} "
              f"struct.bbox={r['struct']['bbox']} n_cand={r['struct']['n_cand']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n结果已保存: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())