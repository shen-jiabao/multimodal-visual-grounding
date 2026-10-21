"""FINAL: Faster R-CNN + 启发式视觉 Grounding（2.1 img/s）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)

DATA_ROOT = Path(r"D:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理")
RESULT_PATH = Path(r"D:\Desktop\2026AiC\result.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_IMG_SIZE = 500
BBOX_PADDING = 0.80
MIN_AREA = 0.04
MAX_AREA = 0.65

print(f"[INFO] 设备: {DEVICE}")
print(f"[INFO] 图像尺寸: {MAX_IMG_SIZE}px")

COCO_CATEGORIES = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

QUERY_TO_COCO = {
    "person": ["person", "man", "woman", "people", "girl", "boy", "guy", "lady", "child", "kid", "walking", "standing", "sitting", "rider"],
    "car": ["car", "vehicle", "automobile", "sedan"],
    "bicycle": ["bicycle", "bike", "cyclist"],
    "motorcycle": ["motorcycle", "motorbike"],
    "bus": ["bus"],
    "truck": ["truck", "van"],
    "airplane": ["airplane", "plane"],
    "boat": ["boat", "ship"],
    "umbrella": ["umbrella", "parasol"],
    "chair": ["chair", "armchair"],
    "couch": ["couch", "sofa", "seat"],
    "bench": ["bench"],
    "dining table": ["table", "dining table"],
    "bed": ["bed"],
    "tv": ["tv", "television", "screen", "monitor"],
    "laptop": ["laptop", "notebook", "computer"],
    "cell phone": ["phone", "cell phone", "mobile phone"],
    "microwave": ["microwave"],
    "oven": ["oven"],
    "refrigerator": ["refrigerator", "fridge"],
    "sink": ["sink"],
    "bottle": ["bottle", "water bottle"],
    "cup": ["cup", "mug", "glass"],
    "bowl": ["bowl", "plate"],
    "pizza": ["pizza"],
    "cake": ["cake"],
    "clock": ["clock"],
    "book": ["book"],
    "traffic light": ["traffic light", "traffic signal"],
    "stop sign": ["stop sign", "sign"],
    "backpack": ["backpack", "bag"],
    "handbag": ["handbag", "purse"],
    "suitcase": ["suitcase", "luggage"],
    "tie": ["tie"],
    "sports ball": ["ball", "football", "basketball"],
    "kite": ["kite"],
    "skateboard": ["skateboard", "skater"],
    "surfboard": ["surfboard"],
    "frisbee": ["frisbee"],
    "bird": ["bird"],
    "cat": ["cat", "kitten"],
    "dog": ["dog", "puppy"],
    "horse": ["horse"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle"],
    "potted plant": ["plant", "potted plant", "flower"],
    "remote": ["remote"],
    "baseball bat": ["bat", "baseball bat"],
}

KEYWORD_TO_CATNAME = {}
for _cat_name, _kws in QUERY_TO_COCO.items():
    for _kw in _kws:
        KEYWORD_TO_CATNAME[_kw] = _cat_name

SPATIAL_X_RANGES = [
    ("leftmost", (0.00, 0.15)),
    ("left", (0.00, 0.40)),
    ("rightmost", (0.85, 1.00)),
    ("right", (0.60, 1.00)),
    ("center", (0.40, 0.60)),
    ("middle", (0.35, 0.65)),
    ("front", (0.40, 0.60)),
]

SPATIAL_Y_RANGES = [
    ("top", (0.00, 0.35)),
    ("upper", (0.00, 0.40)),
    ("bottom", (0.60, 1.00)),
    ("lower", (0.60, 1.00)),
    ("above", (0.00, 0.55)),
    ("foreground", (0.55, 1.00)),
    ("closest", (0.55, 1.00)),
]

ORDINAL_POSITIONS = [
    ("first", 0.15),
    ("second", 0.35),
    ("third", 0.55),
    ("fourth", 0.75),
    ("fifth", 0.88),
]


def parse_query(query: str) -> Dict:
    q = query.lower()
    sx, sy = [], []
    ordinal = None
    coco_cats = []
    for sw, rng in SPATIAL_X_RANGES:
        if sw in q:
            sx.append(rng)
    for sw, rng in SPATIAL_Y_RANGES:
        if sw in q:
            sy.append(rng)
    for ord_name, pos in ORDINAL_POSITIONS:
        if ord_name in q:
            ordinal = pos
            break
    for w in q.replace(",", " ").replace(".", " ").split():
        if w in KEYWORD_TO_CATNAME:
            cat = KEYWORD_TO_CATNAME[w]
            if cat not in coco_cats:
                coco_cats.append(cat)
    return {"spatial_x": sx, "spatial_y": sy, "ordinal": ordinal, "coco_cats": coco_cats}


def enforce_area(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area < MIN_AREA:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tw = max(x2 - x1, np.sqrt(MIN_AREA) * 1.3)
        th = max(y2 - y1, np.sqrt(MIN_AREA) / 1.3)
        x1, y1 = max(0, cx - tw / 2), max(0, cy - th / 2)
        x2, y2 = min(1.0, x1 + tw), min(1.0, y1 + th)
        return np.array([x1, y1, x2, y2])
    elif area > MAX_AREA:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tw, th = np.sqrt(MAX_AREA) * 1.2, np.sqrt(MAX_AREA) / 1.2
        x1, y1 = max(0, cx - tw / 2), max(0, cy - th / 2)
        x2, y2 = min(1.0, x1 + tw), min(1.0, y1 + th)
        return np.array([x1, y1, x2, y2])
    return box


def expand_bbox(box_norm: np.ndarray) -> np.ndarray:
    w = box_norm[2] - box_norm[0]
    h = box_norm[3] - box_norm[1]
    expanded = np.array([
        max(0.0, box_norm[0] - w * BBOX_PADDING),
        max(0.0, box_norm[1] - h * BBOX_PADDING),
        min(1.0, box_norm[2] + w * BBOX_PADDING),
        min(1.0, box_norm[3] + h * BBOX_PADDING),
    ])
    return enforce_area(expanded)


def score_boxes_fast(boxes_norm: np.ndarray, scores_raw: np.ndarray,
                     labels_int: np.ndarray, parsed: Dict) -> int:
    n = len(scores_raw)
    if n == 0:
        return 0
    cat_bonus = np.zeros(n)
    if parsed["coco_cats"]:
        for i in range(n):
            if COCO_CATEGORIES[labels_int[i]] in parsed["coco_cats"]:
                cat_bonus[i] = 4.0
    cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
    cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
    spatial_bonus = np.zeros(n)
    for rng in parsed["spatial_x"]:
        in_range = (cx >= rng[0]) & (cx <= rng[1])
        spatial_bonus[in_range] += 2.0
        out = ~in_range
        if out.any():
            dists = np.minimum(np.abs(cx[out] - rng[0]), np.abs(cx[out] - rng[1]))
            spatial_bonus[out] -= dists * 2.0
    for rng in parsed["spatial_y"]:
        in_range = (cy >= rng[0]) & (cy <= rng[1])
        spatial_bonus[in_range] += 2.0
        out = ~in_range
        if out.any():
            dists = np.minimum(np.abs(cy[out] - rng[0]), np.abs(cy[out] - rng[1]))
            spatial_bonus[out] -= dists * 2.0
    ordinal_bonus = np.zeros(n)
    if parsed["ordinal"] is not None:
        target_x = parsed["ordinal"]
        dist = np.abs(cx - target_x)
        ordinal_bonus[dist < 0.12] += 3.0
        ordinal_bonus -= dist * 4.0
    total = scores_raw * 1.5 + cat_bonus + spatial_bonus + ordinal_bonus
    return int(np.argmax(total))


def heuristic_fast(boxes_norm: np.ndarray, img_bgr: np.ndarray, parsed: Dict) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    rows, cols = 8, 8
    cell_h = h // rows
    cell_w = w // cols
    grid_scores = np.zeros((rows, cols))
    for rng in parsed["spatial_x"]:
        for c in range(cols):
            cx = (c + 0.5) / cols
            if rng[0] <= cx <= rng[1]:
                grid_scores[:, c] += 1.0
    for rng in parsed["spatial_y"]:
        for r in range(rows):
            cy = (r + 0.5) / rows
            if rng[0] <= cy <= rng[1]:
                grid_scores[r, :] += 1.0
    if parsed["ordinal"] is not None:
        target_col = int(parsed["ordinal"] * cols)
        for c in range(cols):
            weight = max(0, 1.5 - abs(c - target_col) / cols * 4)
            grid_scores[:, c] += weight
    if len(boxes_norm) > 0:
        for box_norm in boxes_norm:
            b_cx = (box_norm[0] + box_norm[2]) / 2
            b_cy = (box_norm[1] + box_norm[3]) / 2
            for r in range(rows):
                for c in range(cols):
                    gx = (c + 0.5) / cols
                    gy = (r + 0.5) / rows
                    dist = np.sqrt((gx - b_cx) ** 2 + (gy - b_cy) ** 2)
                    grid_scores[r, c] += max(0, 0.8 - dist * 2.5)
    if cell_h > 0 and cell_w > 0:
        edges = cv2.Canny(img_bgr, 80, 180).astype(np.float32) / 255.0
        for r in range(rows):
            for c in range(cols):
                cell = edges[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                grid_scores[r, c] += cell.mean() * 1.2
    if grid_scores.max() < 0.05:
        for r in range(rows):
            for c in range(cols):
                gx = (c + 0.5) / cols - 0.5
                gy = (r + 0.5) / rows - 0.6
                grid_scores[r, c] = max(0, 0.3 - np.sqrt(gx**2 + gy**2))
    flat_scores = grid_scores.flatten()
    top_k = max(8, int(len(flat_scores) * 0.3))
    top_indices = np.argpartition(-flat_scores, top_k)[:top_k]
    best_idx = int(np.argmax(flat_scores))
    best_r, best_c = best_idx // cols, best_idx % cols
    best_cx = (best_c + 0.5) / cols
    best_cy = (best_r + 0.5) / rows
    all_x1 = [idx % cols / cols for idx in top_indices]
    all_y1 = [idx // cols / cols for idx in top_indices]
    all_x2 = [(idx % cols + 1) / cols for idx in top_indices]
    all_y2 = [(idx // cols + 1) / cols for idx in top_indices]
    x1, y1 = min(all_x1), min(all_y1)
    x2, y2 = max(all_x2), max(all_y2)
    bw = min(max(x2 - x1, 0.18), 0.55)
    bh = min(max(y2 - y1, 0.18), 0.55)
    x1 = max(0, best_cx - bw / 2)
    y1 = max(0, best_cy - bh / 2)
    x2 = min(1.0, x1 + bw)
    y2 = min(1.0, y1 + bh)
    return enforce_area(np.array([x1, y1, x2, y2]))


def main(n_test: int = 0):
    print("=" * 60)
    print("加载 Faster R-CNN 模型...")
    print("=" * 60)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(DEVICE)
    print("✓ 模型加载完成")
    to_tensor = T.ToTensor()

    queries_path = DATA_ROOT / "queries" / "queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    qids = list(queries.keys())
    total = len(qids)
    print(f"共 {total} 个样本")

    if RESULT_PATH.exists() and n_test == 0:
        try:
            with open(RESULT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
            done_ids = set(existing.keys()) & set(qids)
            if len(done_ids) > 0:
                print(f"发现已有结果 {len(done_ids)} 条，跳过这些样本...")
                qids = [q for q in qids if q not in done_ids]
                results = existing
                print(f"剩余 {len(qids)} 条待处理")
            else:
                results = {}
        except Exception:
            results = {}
    else:
        results = {}

    if n_test > 0:
        qids = qids[:n_test]
        print(f"测试模式：只跑前 {n_test} 个样本")

    print(f"\n预解析 queries...")
    parsed_all = {qid: parse_query(queries[qid]["query"]) for qid in qids}
    print(f"✓ 完成")

    print(f"\n开始推理...")
    methods = {}
    t0 = time.time()
    last_report = t0
    last_save = t0

    for idx, qid in enumerate(qids):
        qdata = queries[qid]
        parsed = parsed_all[qid]
        img_path = str(DATA_ROOT / qdata["visible"])
        try:
            pil = Image.open(img_path).convert("RGB")
            w, h = pil.size
            scale = min(1.0, MAX_IMG_SIZE / max(w, h))
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                pil = pil.resize((new_w, new_h), Image.BILINEAR)
                w, h = new_w, new_h
            img_np = np.array(pil)
            img_bgr = img_np[:, :, ::-1].copy()
            tensor = to_tensor(pil).to(DEVICE)
            with torch.no_grad():
                det = model([tensor])[0]
            scores = det["scores"].cpu().numpy()
            boxes = det["boxes"].cpu().numpy()
            labels = det["labels"].cpu().numpy()
            keep_mask = scores > 0.25
            scores_f = scores[keep_mask]
            boxes_f = boxes[keep_mask]
            labels_f = labels[keep_mask]
            if len(scores_f) > 0:
                boxes_norm = boxes_f / [w, h, w, h]
                if parsed["coco_cats"]:
                    best_idx = score_boxes_fast(boxes_norm, scores_f, labels_f, parsed)
                    result_box = expand_bbox(boxes_norm[best_idx])
                    method = "rcnn_category"
                else:
                    best_dets = boxes_norm[scores_f > 0.5] if len(scores_f[scores_f > 0.5]) > 0 else boxes_norm[:5]
                    result_box = heuristic_fast(best_dets, img_bgr, parsed)
                    method = "rcnn_heuristic"
            else:
                result_box = heuristic_fast(np.array([]), img_bgr, parsed)
                method = "heuristic"
            methods[method] = methods.get(method, 0) + 1
            results[qid] = {"bbox": result_box.tolist(), "query": qdata["query"], "visible": qdata["visible"]}
        except Exception:
            methods["error"] = methods.get("error", 0) + 1
            results[qid] = {"bbox": [0.25, 0.25, 0.75, 0.75], "query": qdata["query"], "visible": qdata["visible"]}

        processed = idx + 1
        now = time.time()
        if now - last_report > 5.0 or processed == len(qids):
            elapsed = now - t0
            speed = processed / elapsed
            eta = (len(qids) - processed) / speed
            print(f"  [{processed}/{len(qids)}] speed={speed:.2f} img/s, elapsed={elapsed:.0f}s, eta={eta:.0f}s")
            last_report = now

        if now - last_save > 60.0 or processed == len(qids):
            with open(RESULT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            last_save = now

    elapsed = time.time() - t0
    print(f"\n✓ 推理完成! 总耗时 {elapsed/60:.1f} 分钟, 速度={len(qids)/elapsed:.2f} img/s")
    print(f"\n方法统计:")
    for m, cnt in methods.items():
        pct = 100 * cnt / len(qids)
        print(f"  {m:20s}: {cnt:4d} ({pct:.1f}%)")
    bboxes = np.array([r["bbox"] for r in results.values() if r["bbox"]])
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    print(f"\nbbox 面积: mean={areas.mean():.3f}, median={np.median(areas):.3f}, min={areas.min():.3f}, max={areas.max():.3f}")
    print(f"\n保存到 {RESULT_PATH}...")
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✓ 完成！共 {len(results)} 条预测结果")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(n_test=int(sys.argv[1]))
    else:
        main(n_test=0)
