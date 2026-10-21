# -*- coding: utf-8 -*-
"""序数三连 A/B 对比图渲染（纯 CPU，不加载模型）。

对同一张可见光图 000023.png 的三个序数 query，叠加两套框：
  红色框 = GD origin 轨（detect=select=原句，与锁定基线 IoU=1.0，合并大框）
  绿色框 = LocateAnything L2（high-res slow，物体尺度窄框）

框坐标均为归一化 xyxy [x1,y1,x2,y2]，渲染时按原图宽高换算像素。
输出到 viz_out/{qid}_gd_vs_l2.png。
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import cv2
import numpy as np

DATA_ROOT = r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理"
TEST_JSON = os.path.join(DATA_ROOT, "test.json")
RUNNER_JSON = r"d:\Desktop\2026AiC\variants_gd_30.json"
OUT_DIR = r"d:\Desktop\2026AiC\viz_out"

QIDS = ["000023_002", "000023_003", "000023_004"]

# 远端 predictions_v2_highres_slow.json 中三条的 Locate L2 框（归一化 xyxy）
L2_BOXES: Dict[str, List[float]] = {
    "000023_002": [0.084, 0.788, 0.124, 0.864],
    "000023_003": [0.135, 0.774, 0.174, 0.849],
    "000023_004": [0.136, 0.774, 0.174, 0.848],
}

# BGR 配色
RED = (0, 0, 255)
GREEN = (0, 200, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def to_pixel(box: List[float], w: int, h: int) -> tuple:
    """归一化 xyxy 转像素整数坐标，并夹到图像边界内。"""
    x1, y1, x2, y2 = box
    p = (max(0, int(round(x1 * w))), max(0, int(round(y1 * h))),
         min(w - 1, int(round(x2 * w))), min(h - 1, int(round(y2 * h))))
    return p


def draw_labeled_box(img: np.ndarray, box_px: tuple, color: tuple,
                     tag: str) -> None:
    """在图上画粗框，并在框上方绘制带底色的文字标签。"""
    x1, y1, x2, y2 = box_px
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    ty = max(th + 4, y1 - 6)
    cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 8, ty + 2), color, -1)
    cv2.putText(img, tag, (x1 + 4, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


def box_area(box: List[float]) -> float:
    """计算归一化框面积。"""
    return float((box[2] - box[0]) * (box[3] - box[1]))


def imread_unicode(path: str) -> np.ndarray:
    """兼容 Windows 中文路径读图（cv2.imread 不支持非 ASCII 路径）。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """兼容 Windows 中文路径写图。"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise OSError(f"encode failed: {path}")
    buf.tofile(path)


def main() -> None:
    """渲染三张序数 A/B 对比图。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    test_data = json.load(open(TEST_JSON, encoding="utf-8"))
    runner = json.load(open(RUNNER_JSON, encoding="utf-8"))

    for qid in QIDS:
        rec = test_data[qid]
        img_bgr = imread_unicode(os.path.join(DATA_ROOT, rec["visible"]))
        h, w = img_bgr.shape[:2]

        gd_box = runner[qid]["origin"]["bbox"]
        l2_box = L2_BOXES[qid]

        # 先画 L2（绿）再画 GD（红），两框不重叠，顺序无遮挡问题
        draw_labeled_box(img_bgr, to_pixel(l2_box, w, h), GREEN,
                         f"Locate L2 (area={box_area(l2_box):.3f})")
        draw_labeled_box(img_bgr, to_pixel(gd_box, w, h), RED,
                         f"GD origin (area={box_area(gd_box):.3f})")

        # 顶部信息条：query + 图例
        banner_h = 70
        banner = np.full((banner_h, w, 3), 30, dtype=np.uint8)
        cv2.putText(banner, f"{qid}  {rec['query']}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        cv2.rectangle(banner, (12, 44), (34, 60), RED, -1)
        cv2.putText(banner, "GD origin (merged)", (40, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        gx = 300
        cv2.rectangle(banner, (gx, 44), (gx + 22, 60), GREEN, -1)
        cv2.putText(banner, "Locate L2 (object-scale)", (gx + 28, 57),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        out = np.vstack([banner, img_bgr])

        out_path = os.path.join(OUT_DIR, f"{qid}_gd_vs_l2.png")
        imwrite_unicode(out_path, out)
        print(f"saved {out_path}  (img {w}x{h})")


if __name__ == "__main__":
    main()