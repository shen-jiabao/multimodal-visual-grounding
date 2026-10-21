"""文本匹配与空间约束解析模块。

三级匹配链：精确 label 匹配 -> 双向单词子集软匹配 -> 顺序子串匹配，
再按空间/序数约束排序候选框；深度类约束（closest/farthest）走深度辅助分支。
"""
from __future__ import annotations

import re

from src.postprocess.bbox import normalize_text

_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

_FIRST_BLACKLIST = {"aid", "floor", "class", "place", "time", "base", "hand"}

_STOPWORDS = {"the", "a", "an", "of", "on", "in", "to", "from", "and", "or"}


def flatten_detections(dets: list[dict]) -> list[dict]:
    """
    展平 predict_boxes 分块结果 -> [{label, bbox_pixel, score}, ...]
    predict_boxes 返回 {"scores", "labels", "boxes"}，无 "label" 键。
    bbox_pixel 为像素坐标 xyxy。
    """
    out = []
    for d in dets:
        for lab, box, sc in zip(d["labels"], d["boxes"], d["scores"]):
            out.append({
                "label": str(lab),
                "bbox_pixel": box,
                "score": float(sc),
            })
    return out


def _words(text: str) -> set[str]:
    return set(normalize_text(text).split())


def _word_level_substring(a: str, b: str) -> bool:
    """双向单词子集 + 双约束：词数比 <=2 且绝对差 <=3。"""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False

    if wa.issubset(wb):
        shorter, longer = wa, wb
    elif wb.issubset(wa):
        shorter, longer = wb, wa
    else:
        return False

    if shorter.issubset(_STOPWORDS):
        return False

    return len(longer) <= 2 * len(shorter) + 2 and (len(longer) - len(shorter)) <= 3


def _sequential_substring(label: str, query: str) -> bool:
    """顺序子串：标签词按顺序出现在查询中（允许插入修饰词）。"""
    q_words = normalize_text(query).split()
    l_words = normalize_text(label).split()
    if not l_words or not q_words:
        return False
    if not set(l_words) - _STOPWORDS:
        return False

    it = iter(q_words)
    return all(w in it for w in l_words)


def parse_spatial_constraint(query: str) -> dict | None:
    """解析空间/序数约束。

    支持：first/second/... from the left/right/top/bottom；
    from left to right, the first ...；leftmost/rightmost/topmost/bottommost；
    closest/nearest/farthest/furthest（kind="depth"）。
    """
    q = normalize_text(query)

    # 黑名单必须**词边界**匹配。原先用朴素子串 `f"first {blocked}" in q`，
    # 会把 "first handrail from the left" 里的 "first hand" 当黑名单删掉，
    # 剩 "rail from the left" → 序数丢失、约束静默降级为 None。
    for blocked in _FIRST_BLACKLIST:
        q = re.sub(rf"\bfirst\s+{blocked}\b", "", q)

    ord_pat = "|".join(_ORDINALS)

    # 桥（序数与 from 之间的定语）先按原式匹配；原式只允许「纯单词 + 空白」，
    # 于是 "first white diamond-shaped sign from bottom to top" 这类带连字符 /
    # 逗号 / 撇号的定语整条掉出解析（实测 22 条）。
    m = re.search(
        rf"\b({ord_pat})\b\s+(?:\w+\s+)*?from\s+(?:the\s+)?(left|right|top|bottom)",
        q,
    )
    # 放宽桥：仅在原式未命中时启用，故对原本能解析的 query **逐一等价**
    # （group(1)/group(2) 是同样的词，返回的 axis/direction/rank 完全相同），
    # 对全量是严格超集——不可能让任何已解析的 query 失败或改判。
    if not m:
        m = re.search(
            rf"\b({ord_pat})\b[\s\w,.'\-]{{0,90}}?from\s+(?:the\s+)?"
            r"(left|right|top|bottom)",
            q,
        )
    if m:
        return {
            "axis": "x" if m.group(2) in ("left", "right") else "y",
            "direction": 1 if m.group(2) in ("left", "top") else -1,
            "rank": _ORDINALS[m.group(1)],
        }

    m = re.search(
        rf"from\s+(left|right)\s+to\s+(right|left)[,.]?\s+(?:the\s+)?({ord_pat})\b",
        q,
    )
    if m:
        return {
            "axis": "x",
            "direction": 1 if m.group(1) == "left" else -1,
            "rank": _ORDINALS[m.group(3)],
        }

    m = re.search(r"\b(leftmost|rightmost|topmost|bottommost)\b", q)
    if m:
        return {
            "axis": "x" if m.group(1) in ("leftmost", "rightmost") else "y",
            "direction": 1 if m.group(1) in ("leftmost", "topmost") else -1,
            "rank": 1,
        }

    m = re.search(r"\b(closest|nearest|farthest|furthest)\b", q)
    if m:
        is_near = m.group(1) in ("closest", "nearest")
        return {
            "axis": "y",
            "direction": 1 if is_near else -1,
            "rank": 1,
            "kind": "depth",
        }

    return None


def match_by_position(query: str, detections: list[dict]) -> dict | None:
    """三级匹配链：精确 -> 双向子集 -> 顺序子串 -> 空间排序 -> 最高分。"""
    q_norm = normalize_text(query)

    cands = [d for d in detections if normalize_text(d["label"]) == q_norm]

    if not cands:
        cands = [
            d for d in detections
            if _word_level_substring(d["label"], query)
            or _word_level_substring(query, d["label"])
        ]

    if not cands:
        cands = [
            d for d in detections
            if _sequential_substring(d["label"], query)
            or _sequential_substring(query, d["label"])
        ]

    if not cands:
        return None

    cons = parse_spatial_constraint(query)

    if cons and cons.get("kind") == "depth":
        return max(cands, key=lambda d: d["score"])

    if cons and len(cands) > 1:
        axis = 0 if cons["axis"] == "x" else 1

        def key(d):
            box = d["bbox_pixel"]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            return cx if axis == 0 else cy

        cands.sort(key=key, reverse=(cons["direction"] < 0))
        rank = min(max(cons["rank"], 1), len(cands))
        return cands[rank - 1]

    return max(cands, key=lambda d: d["score"])


def match_query_with_depth(
    query: str,
    detections: list[dict],
    depth_info: dict | None,
) -> dict | None:
    """位置优先 + 深度辅助：仅 closest/farthest 且深度 high/medium 时排序。"""
    cons = parse_spatial_constraint(query)

    if (
        cons
        and cons.get("kind") == "depth"
        and depth_info
        and depth_info.get("confidence") in ("high", "medium")
    ):
        cands = [
            d for d in detections
            if normalize_text(d["label"]) == normalize_text(query)
            or _word_level_substring(d["label"], query)
            or _sequential_substring(d["label"], query)
        ]

        if len(cands) > 1:
            valid = depth_info["valid_mask"]
            norm = depth_info["norm_lin"]

            def dkey(d):
                b = d["bbox_pixel"]
                x1, y1, x2, y2 = map(int, b)
                roi = norm[y1:y2, x1:x2]
                m = valid[y1:y2, x1:x2]
                vals = roi[m]
                return float(vals.mean()) if vals.size else 0.5

            cands.sort(key=dkey, reverse=(cons["direction"] < 0))
            return cands[0]

    return match_by_position(query, detections)
