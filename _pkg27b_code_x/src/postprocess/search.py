"""候选搜索与聚合：generate-and-verify 的 verify 层（纯函数，不依赖 torch）。

零样本（不训练、不伪标注）约束下，单次贪心解码只给出一个框，代码里没有任何
"再挑一次"的余地。搜索式推理的前提是：让模型对同一 query 产出**多个候选框**，
再用与标注无关的一致性信号（候选之间的 IoU 投票）选出最可信的一个，并把
**票数占比**当作置信度回填。

本模块只做纯计算（坐标域不做假设，像素或 [0,1] 均可，IoU 与域无关）：

    - :func:`iou_xyxy`：两个 xyxy 框的 IoU；
    - :func:`cluster_by_iou`：贪心 IoU 聚类（簇代表为簇内坐标均值）；
    - :func:`aggregate_candidates`：投票选簇 + 簇内加权平均  单框 + 置信度。

修复了原先 "scores = default_score(1.0) 恒定" 导致 matcher 里
``max(cands, key=score)`` 退化为任取一个的问题。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

Box = Sequence[float]


def iou_xyxy(a: Box, b: Box) -> float:
    """两个 xyxy 框的交并比；并集面积为 0 时返回 0.0。"""
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def cluster_by_iou(boxes: Sequence[Box],
                   threshold: float = 0.5) -> List[List[int]]:
    """贪心 IoU 聚类：与任一簇代表 IoU >= threshold 则并入该簇（先到先得）。

    簇代表为该簇已有成员坐标的算术均值（并入后更新）。返回各簇的成员下标列表，
    元素顺序与 ``boxes`` 顺序一致（确定性，无随机性）。
    """
    clusters: List[List[int]] = []
    reps: List[List[float]] = []
    for i, box in enumerate(boxes):
        b = [float(v) for v in box[:4]]
        placed = False
        for ci, rep in enumerate(reps):
            if iou_xyxy(b, rep) >= threshold:
                clusters[ci].append(i)
                n = float(len(clusters[ci]))
                reps[ci] = [(rep[k] * (n - 1) + b[k]) / n for k in range(4)]
                placed = True
                break
        if not placed:
            clusters.append([i])
            reps.append(b)
    return clusters


def aggregate_candidates(candidates: Sequence[Box],
                         weights: Optional[Sequence[float]] = None,
                         threshold: float = 0.5,
                         min_votes: int = 1) -> Optional[Dict[str, object]]:
    """投票选簇 + 簇内加权平均（搜索的"选择"步骤）。

    Args:
        candidates: 候选框序列（xyxy，同一 query 的多次采样/多次提问结果）。
        weights: 与 ``candidates`` 等长的权重（默认全 1.0）。
        threshold: 判定"指向同一目标"的 IoU 阈值。
        min_votes: 最大簇的票数低于该值时返回 None（调用方据此退回单候选/兜底）。

    Returns:
        ``{"box", "votes", "total", "confidence", "weight"}``：

        - ``box``：最大簇的加权平均框（已规整为 x1<=x2、y1<=y2）；
        - ``votes`` / ``total``：该簇票数 / 候选总数；
        - ``confidence = 簇内权重和 / 全部权重和``，可直接作为预测置信度。
    """
    if not candidates:
        return None
    boxes = [[float(v) for v in c[:4]] for c in candidates]
    if weights is None:
        wts = [1.0] * len(boxes)
    else:
        wts = [float(w) for w in weights]
        if len(wts) != len(boxes):
            raise ValueError("weights 与 candidates 长度不一致")

    clusters = cluster_by_iou(boxes, threshold)

    # 选簇判据 = (票数, 权重和)：先比同意样本数，再比权重和
    def _rank(ci: int) -> tuple:
        idx = clusters[ci]
        return (len(idx), sum(wts[i] for i in idx))

    best = max(range(len(clusters)), key=_rank)
    idx = clusters[best]
    wsum = sum(wts[i] for i in idx)
    if len(idx) < int(min_votes) or wsum <= 0:
        return None

    merged = [sum(boxes[i][k] * wts[i] for i in idx) / wsum for k in range(4)]
    x1, y1, x2, y2 = merged
    total = sum(wts)
    return {
        "box": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        "votes": len(idx),
        "total": len(boxes),
        "confidence": wsum / total if total > 0 else 0.0,
        "weight": wsum,
    }