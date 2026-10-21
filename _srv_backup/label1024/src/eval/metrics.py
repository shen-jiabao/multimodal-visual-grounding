"""评测指标模块。

实现赛题评测指标 ACC@0.5：
    预测框与真实框 IoU >= 0.5 视为定位正确；
    ACC@0.5 = 正确样本数 / 总样本数。

输入契约：
    - 框为 [x1,y1,x2,y2] 坐标；反向坐标（x2<x1 或 y2<y1）、非有限值、
      长度 != 4 一律抛 ValueError（评测不接受脏输入）；
    - None 表示“无预测”（来自 predictor 的漏检），IoU 记 0、计为错误样本；
    - 空输入返回 NaN，表示无数据可评，避免被误读为性能 0。
"""
from __future__ import annotations

import math
from typing import Sequence

__all__ = ["iou", "acc_at_iou", "mean_iou", "summarize"]


def _validated_box(box: Sequence[float], name: str) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"{name} 需要 4 个坐标: {box}")
    vals = [float(v) for v in box]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name} 含 NaN/Inf: {box}")
    x1, y1, x2, y2 = vals
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{name} 为反向坐标: {box}")
    return x1, y1, x2, y2


def iou(box_a: Sequence[float] | None, box_b: Sequence[float] | None) -> float:
    """计算两个 [x1,y1,x2,y2] 框的交并比（IoU）。

    None 视为无预测，IoU 记 0。反向/非有限/长度错误的框抛 ValueError。
    """
    if box_a is None or box_b is None:
        return 0.0
    xa1, ya1, xa2, ya2 = _validated_box(box_a, "box_a")
    xb1, yb1, xb2, yb2 = _validated_box(box_b, "box_b")

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def acc_at_iou(preds: Sequence[Sequence[float] | None],
               gts: Sequence[Sequence[float]],
               threshold: float = 0.5) -> float:
    """计算 ACC@threshold。

    None 预测视为错误样本；长度不一致抛 ValueError；空输入返回 NaN。

    Args:
        preds: 预测框列表（元素可为 None 表示无预测）。
        gts: 真实框列表（与 preds 一一对应）。
        threshold: IoU 阈值，默认 0.5。

    Returns:
        0~1 的准确率。
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold 必须在 [0,1]: {threshold}")
    if len(preds) != len(gts):
        raise ValueError(f"preds 与 gts 数量不一致: {len(preds)} vs {len(gts)}")
    if len(preds) == 0:
        return float("nan")
    correct = sum(1 for p, g in zip(preds, gts) if iou(p, g) >= threshold)
    return correct / len(preds)


def mean_iou(preds: Sequence[Sequence[float] | None],
             gts: Sequence[Sequence[float]]) -> float:
    """平均 IoU（辅助指标，用于更细致的性能分析）。

    None 预测视为 IoU 0；长度不一致抛 ValueError；空输入返回 NaN。
    """
    if len(preds) != len(gts):
        raise ValueError(f"preds 与 gts 数量不一致: {len(preds)} vs {len(gts)}")
    if len(preds) == 0:
        return float("nan")
    total = sum(iou(p, g) for p, g in zip(preds, gts))
    return total / len(preds)


def summarize(preds: Sequence[Sequence[float] | None],
              gts: Sequence[Sequence[float]],
              threshold: float = 0.5) -> dict[str, float | int]:
    """汇总评测报告：ACC、mean IoU 与无预测统计。

    Returns:
        {"acc", "mean_iou", "num_total", "num_none", "num_correct"}
        空输入时 acc/mean_iou 为 NaN。
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold 必须在 [0,1]: {threshold}")
    if len(preds) != len(gts):
        raise ValueError(f"preds 与 gts 数量不一致: {len(preds)} vs {len(gts)}")
    n = len(preds)
    num_none = sum(1 for p in preds if p is None)
    num_correct = sum(
        1 for p, g in zip(preds, gts) if p is not None and iou(p, g) >= threshold
    )
    total_iou = sum(iou(p, g) for p, g in zip(preds, gts))
    return {
        "acc": num_correct / n if n else float("nan"),
        "mean_iou": total_iou / n if n else float("nan"),
        "num_total": n,
        "num_none": num_none,
        "num_correct": num_correct,
    }
