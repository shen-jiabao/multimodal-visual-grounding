"""序数链提取与单调性修正（跨查询一致性约束，零样本可用）。

背景（离线实测，见 outputs/复赛/chain/ordinal_chains_20260927.json）：
本数据集的 query 中存在大量 "the <k-th> <同一物体> from the left/right/top/bottom"
式的**序数链**。同一张图 + 同一物体 + 同一轴向/方向的链，各成员框的中心坐标
必须随序数单调。实测 181 条有效链里约 30% 违反该约束（A_ratio = 0.7027），
且多数并非"位置不准"，而是把两个序数的框**写反或写到别处**这类错误可以用
纯几何修正，不需要标注、不需要训练。

本模块提供：

    - :func:`residual_tokens` / :func:`chain_of`：按「残余词 + 轴向 + 方向」把
      query 归到同一条链（残余词 = 去掉序数词/方位词/停用词后的实义词集合，
      保证同链成员是**同一类物体**，故其框可互换）；
    - :func:`classify_chain`：链状态分类（ordered / A / B_dup / B_tie / B_broken），
      判据与 ``scripts/analyze_ordinal_chains.py`` 完全一致，保证度量口径可比；
    - :func:`enforce_ordinal_chains`：**排序式重指派**（assignment search）。
      同链成员是同类物体，位置排序后按序数重新指派即 L1-最优的序数分配；
      已单调的链完全不变（幂等），只动被破坏的链。

坐标域：本模块统一处理 **[0,1] 归一化 xyxy**（predictor 归一化之后调用）。
"""
from __future__ import annotations

import re

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.inference.matcher import (
    _ORDINALS,
    normalize_text,
    parse_spatial_constraint,
)
from src.postprocess.search import iou_xyxy

# 与 scripts/analyze_ordinal_chains.py 保持一致的词表（度量口径统一）
STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "with", "and", "to", "from", "at", "is",
    "are", "by", "for", "that", "this", "it", "its", "as", "or", "be", "was",
    "there", "which", "who", "his", "her", "their", "has", "have", "having",
    "not", "no", "all", "also", "than", "over", "under", "near", "beside",
}
AXIS_WORDS = {"left", "right", "top", "bottom", "from", "to", "the", "most"}
ORD_EXTRA = {"leftmost", "rightmost", "topmost", "bottommost"}
EPS = 0.02


def residual_tokens(query: str) -> frozenset:
    """query 去掉序数词/方位词/停用词后的实义词集合（同链成员的"物体身份"）。"""
    t = normalize_text(query)
    words = [re.sub(r"[^a-z0-9]+", "", w) for w in t.split()]
    return frozenset(
        w for w in words
        if w and w not in STOPWORDS
        and w not in _ORDINALS and w not in ORD_EXTRA and w not in AXIS_WORDS
    )


def chain_of(query: str) -> Optional[dict]:
    """从单条 query 解析出链成员信息；非序数链返回 None。

    Returns:
        ``{"axis": "x"|"y", "direction": 1|-1, "rank": int, "key": frozenset}``
        （direction=1 表示 rank 越大坐标越大，如 "first ... from the left"）。
    """
    cons = parse_spatial_constraint(query)
    if not cons or cons.get("kind") == "depth" or int(cons.get("rank", 0)) <= 0:
        return None
    key = residual_tokens(query)
    if not key:
        return None
    return {
        "axis": cons["axis"],
        "direction": int(cons["direction"]),
        "rank": int(cons["rank"]),
        "key": key,
    }


def classify_chain(items: Sequence[tuple], eps: float = EPS) -> str:
    """链状态分类（与 analyze_ordinal_chains.py 同判据）。

    Args:
        items: ``[(qid, query, box, cons), ...]``，``cons`` 为
            :func:`src.inference.matcher.parse_spatial_constraint` 的结果。

    Returns:
        ``skip_axis`` / ``skip_rank`` / ``ordered`` / ``A`` / ``B_dup`` /
        ``B_tie`` / ``B_broken``。其中 ``A`` 表示"完全无解释的破序"（收益上限），
        ``B_*`` 表示可用重复/并列/断裂解释的破序。
    """
    axes = {c["axis"] for _, _, _, c in items}
    dirs = {c["direction"] for _, _, _, c in items}
    ranks = sorted(c["rank"] for _, _, _, c in items)
    k = len(items)
    if len(axes) != 1 or len(dirs) != 1:
        return "skip_axis"
    if ranks != list(range(1, k + 1)):
        return "skip_rank"
    axis = axes.pop()
    d = dirs.pop()
    ordered_items = sorted(items, key=lambda x: x[3]["rank"])
    xs = [(box[0] + box[2]) / 2 if axis == "x" else (box[1] + box[3]) / 2
          for _, _, box, _ in ordered_items]
    if all((xs[i + 1] - xs[i]) * d > 0 for i in range(k - 1)):
        return "ordered"
    for i in range(k):
        for j in range(i + 1, k):
            if iou_xyxy(ordered_items[i][2], ordered_items[j][2]) > 0.5:
                return "B_dup"
    sx = sorted(xs)
    if any(sx[i + 1] - sx[i] <= eps for i in range(k - 1)):
        return "B_tie"
    if k >= 3:
        gaps = np.diff(np.asarray(sx, dtype=np.float64))
        med = float(np.median(gaps))
        if med > 0 and float(gaps.max()) > 3.0 * med:
            return "B_broken"
    return "A"


def _center(box: Sequence[float], axis_i: int) -> float:
    return (float(box[0]) + float(box[2])) / 2 if axis_i == 0 \
        else (float(box[1]) + float(box[3])) / 2


def _shift_box(box: Sequence[float], delta: float, axis_i: int) -> List[float]:
    """沿轴平移框，并在 [0,1] 内裁剪（优先保住框的尺寸）。"""
    b = [float(v) for v in box[:4]]
    if axis_i == 0:
        w = b[2] - b[0]
        x1 = min(max(b[0] + delta, 0.0), max(0.0, 1.0 - w))
        b[0], b[2] = x1, x1 + w
    else:
        h = b[3] - b[1]
        y1 = min(max(b[1] + delta, 0.0), max(0.0, 1.0 - h))
        b[1], b[3] = y1, y1 + h
    return b


def pava_non_decreasing(values: Sequence[float]) -> List[float]:
    """PAVA（保序回归）：L2 最小的非降序列拟合（mode="project" 使用）。"""
    stack: List[List[float]] = []  # [sum, count]
    for x in values:
        cur = [float(x), 1.0]
        while stack and stack[-1][0] / stack[-1][1] > cur[0] / cur[1]:
            prev = stack.pop()
            cur = [prev[0] + cur[0], prev[1] + cur[1]]
        stack.append(cur)
    out: List[float] = []
    for s, c in stack:
        out.extend([s / c] * int(c))
    return out


def enforce_ordinal_chains(
    queries: Sequence[str],
    boxes: Sequence[Optional[Sequence[float]]],
    mode: str = "sort",
    gap_ratio: float = 0.0,
    max_shift: Optional[float] = None,
) -> List[Optional[List[float]]]:
    """对同一场景内的 query 做序数链单调性修正（排序式重指派）。

    只处理同一次调用内（即同一张图）的 query：链的 key = (轴向, 方向, 残余词集合)，
    要求成员数 >= 2 且 rank 互不相同（同 rank 的重复 query 视为同一目标，取均值
    后一起平移）。

    Args:
        queries: 同图 query 文本（与 ``boxes`` 等长、同序）。
        boxes: 归一化 xyxy 框，缺失用 None 表示（None 不参与、原样返回）。
        mode: ``"sort"`` = 按位置排序后重指派（默认，L1 最优的序数分配）；
            ``"project"`` = PAVA 保序回归（只做最小 L2 微调，不改变指派）。
        gap_ratio: 相邻序数的最小中心间隔 = gap_ratio  链内平均框宽（默认 0.0，
            即已单调的链完全不动；>0 时会把并列/重合的链推开）。
        max_shift: 单框最大平移量（归一化坐标，None = 不限）。

    Returns:
        与 ``boxes`` 等长的新列表（未参与修正的项原样返回）。
    """
    out: List[Optional[List[float]]] = [
        None if b is None else [float(v) for v in b[:4]] for b in boxes
    ]

    groups: Dict[tuple, List[Tuple[int, int]]] = {}
    for i, q in enumerate(queries):
        if out[i] is None:
            continue
        ch = chain_of(q)
        if ch is None:
            continue
        groups.setdefault((ch["axis"], ch["direction"], ch["key"]), []).append(
            (ch["rank"], i)
        )

    for (axis, direction, _key), members in groups.items():
        ranks = sorted({r for r, _ in members})
        if len(ranks) < 2:
            continue
        axis_i = 0 if axis == "x" else 1

        rank_idx = {r: [i for rr, i in members if rr == r] for r in ranks}
        centers = [
            float(np.mean([_center(out[i], axis_i) for i in rank_idx[r]]))
            for r in ranks
        ]
        extents = [
            abs(float(out[i][2]) - float(out[i][0])) if axis_i == 0
            else abs(float(out[i][3]) - float(out[i][1]))
            for _, i in members
        ]
        gap = max(0.0, float(gap_ratio)) * (float(np.mean(extents)) if extents else 0.0)

        sign = 1.0 if direction > 0 else -1.0
        s = [sign * c for c in centers]
        new_s = pava_non_decreasing(s) if mode == "project" else sorted(s)

        if gap > 0:
            for k in range(1, len(new_s)):
                if new_s[k] < new_s[k - 1] + gap:
                    new_s[k] = new_s[k - 1] + gap

        for k, r in enumerate(ranks):
            delta = sign * new_s[k] - centers[k]
            if max_shift is not None:
                delta = max(-max_shift, min(max_shift, delta))
            if abs(delta) <= 1e-12:
                continue
            for i in rank_idx[r]:
                out[i] = _shift_box(out[i], delta, axis_i)

    return out