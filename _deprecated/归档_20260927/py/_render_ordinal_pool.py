# -*- coding: utf-8 -*-
"""渲染方案 2 序数选择验证图（内部诊断，非提交可视化）。

数据源：locate_multi_probe_000023.json 的 P3（count-then-box，含边缘/遮挡实例）。
对 second/third/fourth 各出一张：全部候选为绿色编号框，规则选中的第 N 个为红色粗框。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from locate_ordinal_select import select_nth

DATA_ROOT = r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理"
IMG_REL = r"Images\visible\000023.png"
PROBE_JSON = r"d:\Desktop\2026AiC\locate_multi_probe_000023.json"
OUT_DIR = r"d:\Desktop\2026AiC\viz_out"

# qid -> (序数 rank, query 文案)
TARGETS = [
    ("000023_002", 2, "From left to right, the second stone pier"),
    ("000023_003", 3, "From left to right, the third stone pier"),
    ("000023_004", 4, "From left to right, the fourth stone pier"),
]

GREEN = (0, 200, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def imread_unicode(path: str) -> np.ndarray:
    """兼容中文路径读图。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """兼容中文路径写图。"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise OSError(path)
    buf.tofile(path)


def to_px(box, w, h):
    """归一化 xyxy -> 像素并夹边。"""
    x1, y1, x2, y2 = box
    return (max(0, int(round(x1 * w))), max(0, int(round(y1 * h))),
            min(w - 1, int(round(x2 * w))), min(h - 1, int(round(y2 * h))))


def tag_box(img, box_px, color, tag, thickness, tag_color=WHITE):
    """画框 + 角标。"""
    x1, y1, x2, y2 = box_px
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if tag:
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, tag, (x1 + 4, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tag_color, 2, cv2.LINE_AA)


def main() -> None:
    """对三个序数渲染候选池 + 选中框。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    probe = json.load(open(PROBE_JSON, encoding="utf-8"))
    raw_boxes = probe["P3_count_then_box"]["boxes"]

    # 去重排序后的完整池（所有图共用，因为同一张图）
    _, pool = select_nth(raw_boxes, 1, axis="x", order="asc")
    print(f"去重后候选 {len(pool)} 个（按 cx 升序）：")
    for i, b in enumerate(pool, start=1):
        cx = (b[0] + b[2]) / 2
        print(f"  rank{i}: cx={cx:.3f} box={[round(v,3) for v in b]}")

    base_img = imread_unicode(os.path.join(DATA_ROOT, IMG_REL))
    h, w = base_img.shape[:2]

    for qid, rank, query in TARGETS:
        sel, _ = select_nth(raw_boxes, rank, axis="x", order="asc")
        canvas = base_img.copy()
        # 先画全部候选（绿色细框 + 序号）
        for i, b in enumerate(pool, start=1):
            tag_box(canvas, to_px(b, w, h), GREEN, str(i), 2)
        # 再画选中框（红色粗框）
        tag_box(canvas, to_px(sel, w, h), RED, f"pick #{rank}", 5)

        banner = np.full((56, w, 3), 30, dtype=np.uint8)
        cv2.putText(banner, f"{qid}  {query}", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2, cv2.LINE_AA)
        out = np.vstack([banner, canvas])
        out_path = os.path.join(OUT_DIR, f"{qid}_ordinal_pool.png")
        imwrite_unicode(out_path, out)
        print(f"saved {out_path}  pick={[round(v,3) for v in sel]}")


if __name__ == "__main__":
    main()