# -*- coding: utf-8 -*-
"""通用 bbox 可视化程序：在一张图片上绘制一个或多个归一化框并保存。

既可作为模块 import 使用 draw_boxes()，也可命令行直接调用：

    python viz_boxes.py --image a.png \
        --boxes "0.1,0.2,0.3,0.5;0.4,0.5,0.6,0.8" \
        --labels "obj_a;obj_b" --final "0.4,0.5,0.6,0.8" --out out.png

框坐标为归一化 [x1,y1,x2,y2] ∈ [0,1]，按原图宽高换算到像素。
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

# 默认调色板（BGR），按框顺序循环使用
_DEFAULT_COLORS = [
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 255, 255),   # 黄
    (255, 0, 255),   # 品红
    (255, 255, 0),   # 青
    (0, 165, 255),   # 橙
    (128, 0, 128),   # 紫
    (0, 128, 255),   # 橙红
]
FINAL_COLOR = (0, 0, 255)   # 高亮最终框用红色（BGR）


def _imread_unicode(path: str) -> np.ndarray:
    """兼容中文路径读取图片，返回 BGR。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise FileNotFoundError(f"图片不存在或为空: {path}")
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def draw_boxes(image_bgr: np.ndarray, boxes, labels=None, colors=None,
               final_box=None, thickness: int = 2, font_scale: float = 0.6):
    """在图片副本上绘制归一化框，返回新图（不修改原图）。

    Args:
        image_bgr: HxWx3 BGR 图片。
        boxes: (K,4) 归一化 [x1,y1,x2,y2] 数组或列表。
        labels: 可选，长度 K 的字符串，绘制在框左上角。
        colors: 可选，长度 K 的 (B,G,R) 列表；缺省按调色板循环。
        final_box: 可选，单独高亮的 [x1,y1,x2,y2] 最终框（红色加粗）。
        thickness: 普通框线宽。
        font_scale: 标签字号。
    Returns:
        绘制后的 BGR 图片副本。
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        if colors is not None:
            color = colors[i]
        else:
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
        p1 = (int(round(x1 * w)), int(round(y1 * h)))
        p2 = (int(round(x2 * w)), int(round(y2 * h)))
        cv2.rectangle(img, p1, p2, color, thickness)
        if labels is not None and i < len(labels):
            text = str(labels[i])
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, 1)
            ty = p1[1] - 5 if p1[1] - 5 > th else p1[1] + th + 8
            cv2.rectangle(img, (p1[0], ty - th), (p1[0] + tw, ty + 4),
                          color, -1)
            cv2.putText(img, text, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), 1)
    if final_box is not None:
        x1, y1, x2, y2 = final_box
        cv2.rectangle(img, (int(round(x1 * w)), int(round(y1 * h))),
                      (int(round(x2 * w)), int(round(y2 * h))),
                      FINAL_COLOR, thickness + 2)
    return img


def _parse_boxes(s: str) -> list[list[float]]:
    """解析 "x1,y1,x2,y2;x1,y1,..." 为浮点框列表。"""
    boxes = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [float(v) for v in part.split(",")]
        if len(vals) != 4:
            raise ValueError(f"每个框需 4 个值(x1,y1,x2,y2): {part}")
        boxes.append(vals)
    return boxes


def _parse_final(s: str):
    """解析单个高亮框 "x1,y1,x2,y2"；空串返回 None。"""
    s = s.strip()
    if not s:
        return None
    vals = [float(v) for v in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"高亮框需 4 个值: {s}")
    return vals


def main():
    parser = argparse.ArgumentParser(description="通用 bbox 可视化")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument("--boxes", required=True,
                        help="分号分隔的归一化框 x1,y1,x2,y2;...")
    parser.add_argument("--labels", default="",
                        help="分号分隔的标签（可选，与框一一对应）")
    parser.add_argument("--final", default="",
                        help="高亮最终框 x1,y1,x2,y2（可选）")
    parser.add_argument("--out", required=True, help="输出图片路径")
    args = parser.parse_args()

    img = _imread_unicode(args.image)
    boxes = np.array(_parse_boxes(args.boxes), dtype=np.float64)
    labels = [s for s in args.labels.split(";") if s.strip() != ""] or None
    final_box = _parse_final(args.final)
    out_img = draw_boxes(img, boxes, labels=labels, final_box=final_box)
    cv2.imencode(".png", out_img)[1].tofile(args.out)  # 兼容中文路径
    print(f"已保存: {args.out}（{len(boxes)} 个框"
          + (f"，高亮 1 个" if final_box else "") + "）")


if __name__ == "__main__":
    main()