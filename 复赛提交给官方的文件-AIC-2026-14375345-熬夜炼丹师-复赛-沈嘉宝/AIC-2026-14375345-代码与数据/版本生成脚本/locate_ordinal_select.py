# -*- coding: utf-8 -*-
"""LocateAnything 多候选的确定性序数选择（纯 CPU、无 torch 依赖）。

用途：core 短语让模型一次输出全部同类实例框后，规则按空间轴稳定排序取第 N 个。
设计要点（对应实测）：
- 模型可能重复吐同一批框（P3 把每个框输出两遍）-> 用 IoU 贪心去重，保留首次出现。
- 贴边被截断的实例（如最左石墩）面积极小，绝不能按面积下限滤掉；
  只剔除退化框（坐标非法）与近整图框（area>=0.9）。
- 排序必须确定性：主键中心坐标， tie-break 用另一轴中心，再用原始发射序号兜底，
  避免 set/无序导致同输入不同结果。
"""
from __future__ import annotations

from typing import List, Sequence

# 近整图框视为场景/合并框，不参与"同类实例"序数
JUMBO_AREA = 0.9


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """两个归一化 xyxy 框的 IoU。"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = ((ax2 - ax1) * (ay2 - ay1)
             + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / union) if union > 0 else 0.0


def is_degenerate(box: Sequence[float]) -> bool:
    """退化判定：坐标非法或近整图。贴边小框不算退化。"""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return True
    area = (x2 - x1) * (y2 - y1)
    return area >= JUMBO_AREA


def dedup_boxes(boxes: List[Sequence[float]], iou_thr: float = 0.5
                ) -> List[List[float]]:
    """按发射顺序贪心去重：与任一已保留框 IoU>=阈值则丢弃。

    确定性：不使用 set，保留每个实例在模型输出中的首次出现位置。
    """
    kept: List[List[float]] = []
    for b in boxes:
        b = [float(v) for v in b]
        if is_degenerate(b):
            continue
        if any(iou(b, k) >= iou_thr for k in kept):
            continue
        kept.append(b)
    return kept


def _center(box: Sequence[float]) -> tuple:
    """返回 (cx, cy)。"""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def sort_by_axis(boxes: List[Sequence[float]], axis: str = "x",
                 order: str = "asc") -> List[List[float]]:
    """按中心坐标稳定排序。

    主键 axis（x/y）中心，次键另一轴中心，末位用输入序号兜底，保证全序确定。
    """
    indexed = list(enumerate(boxes))
    if axis == "x":
        keyed = [((_center(b)[0], _center(b)[1], i), b) for i, b in indexed]
    else:
        keyed = [((_center(b)[1], _center(b)[0], i), b) for i, b in indexed]
    keyed.sort(key=lambda t: t[0], reverse=(order == "desc"))
    return [b for _, b in keyed]


def select_nth(boxes: List[Sequence[float]], rank: int, axis: str = "x",
               order: str = "asc", iou_thr: float = 0.5):
    """从原始多候选中选第 rank 个（1 基；rank=-1 取最后）。

    Returns:
        (选中框, 去重排序后的完整候选列表)；无可用候选时为 (None, [])。
    """
    pool = sort_by_axis(dedup_boxes(boxes, iou_thr), axis, order)
    if not pool:
        return None, []
    if rank < 0:
        idx = len(pool) + rank  # -1 -> 最后
    else:
        idx = rank - 1
    idx = int(min(max(idx, 0), len(pool) - 1))
    return pool[idx], pool