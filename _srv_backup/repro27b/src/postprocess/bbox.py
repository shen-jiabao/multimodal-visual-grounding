"""边界框（bbox）处理与校验模块。

包含：
    - 文本归一化（用于查询与检测标签匹配）；
    - 像素坐标 -> 归一化坐标（除以图像宽高，输出 [0,1]）；
    - 预测框合法性校验（反向坐标、越界、NaN、空框）；
    - 坐标裁剪，保证输出符合赛题规范。
"""
from __future__ import annotations

import re
from typing import List

import numpy as np

from src.utils.text import clean_prompt_text


def normalize_text(text: str) -> str:
    """文本归一化：清洗噪声字符、小写、去首尾空白、压缩多余空格、去末尾句点。

    先经 ``clean_prompt_text``（``clean_query`` 的超集：BOM/零宽/连字符族 +
    弯引号/省略号/不换行空格/换行）再做常规归一。label 与 query 走同一规则，
    可修复带 BOM/异体连字符查询的匹配失配，并让**匹配层与提示词层用同一套
    清洗**——提示词里的 query 已被清洗，匹配时若还按原始文本对齐就会错位。
    """
    t = clean_prompt_text(text).lower().rstrip(".")
    return re.sub(r"\s+", " ", t)


def normalize_bbox(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """像素坐标 [x1,y1,x2,y2] 转为归一化坐标（除以宽/高）。

    同时保证 x1<=x2、y1<=y2，并对 [0,1] 越界做裁剪。
    """
    x1, y1, x2, y2 = box[:4]
    x1 = min(max(float(x1), 0.0), width) / width
    x2 = min(max(float(x2), 0.0), width) / width
    y1 = min(max(float(y1), 0.0), height) / height
    y2 = min(max(float(y2), 0.0), height) / height

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def is_valid_bbox(box: List[float], eps: float = 1e-6) -> bool:
    """校验归一化 bbox 是否合法。

    赛题规定：反向坐标（x1>=x2、y1>=y2）、越界、NaN、空框均判无效。
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    if any(np.isnan(v) or np.isinf(v) for v in (x1, y1, x2, y2)):
        return False
    if not (0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and
            0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0):
        return False
    if x2 - x1 <= eps or y2 - y1 <= eps:
        return False
    return True


def sanitize_bbox(box: List[float]) -> List[float]:
    """对预测框做格式修正：裁剪到 [0,1]、修正反向坐标、保证非空。"""
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, y1, x2, y2 = [min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2)]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    # 空框兜底：给一个 1% 大小的小框，避免被判无效
    if x2 - x1 <= 1e-6:
        x2 = min(1.0, x1 + 0.01)
        x1 = max(0.0, x2 - 0.01)
    if y2 - y1 <= 1e-6:
        y2 = min(1.0, y1 + 0.01)
        y1 = max(0.0, y2 - 0.01)
    return [x1, y1, x2, y2]
