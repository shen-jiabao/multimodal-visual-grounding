# -*- coding: utf-8 -*-
"""detect-then-select 纯函数单测（NP 抽取 / 约束解析 / 确定性选择）。

这些测试固化的是**实测踩过的坑**，不是语言学正确性：

  · 序数在方位短语**之前**（``From left to right, the first ...``）时 rank 必须
    是 first 的 1，但 ``the second ... from left to right`` 必须拿到 2 —— 早期
    实现只向后找序数，导致 573 条语序②的 rank 静默退化成 1。
  · 序数嵌在名词短语**中间**，只能多段抹除；把「序数→方位」当连续 span 切掉
    会产出 ``np='The'``（实测 1 条 → 修前 8 条不可用）。
  · 尾部介词清理必须锚定末尾，否则 ``umbrella in the outdoor area`` 被吃成
    ``umbrellaoutdoor`` 这类伪词。
  · 新解析器必须是旧解析器的**超集**：``matcher.parse_spatial_constraint`` 能解
    的，``parse_constraint_ex`` 必须也能解（实测曾出现 11 条退化）。
"""
from __future__ import annotations

import pytest

from src.inference.detect_select import (
    box_centers,
    extract_target,
    is_valid_np,
    parse_constraint_ex,
    select_box,
    select_index,
)
from src.inference.matcher import parse_spatial_constraint


# ------------------------------------------------------------ 约束解析
@pytest.mark.parametrize("query,axis,direction,rank,kind", [
    # 语序①：序数在前，方位在后。注意方向取的是 "from" 后面那个词：
    # "from bottom to top" 的 rank 1 落在图**下**方（y 最大）= direction -1，
    # 与 matcher.parse_spatial_constraint 完全一致。
    ("The first black swan from bottom to top", "y", -1, 1, "pos"),
    ("The third black swan from bottom to top", "y", -1, 3, "pos"),
    ("The first red and white bumper car from right to left", "x", -1, 1, "pos"),
    ("The first double glass entrance doors from the left to right", "x", 1, 1, "pos"),
    # 语序②：方位短语在前，序数在后 —— rank 必须是序数，不能退化成 1
    ("From left to right, the first red letter L", "x", 1, 1, "pos"),
    ("From left to right, the second red letter L", "x", 1, 2, "pos"),
    ("From left to right, the fourth red letter L", "x", 1, 4, "pos"),
    # -est 方位词
    ("The leftmost roadblock", "x", 1, 1, "pos"),
    ("The rightmost roadblock", "x", -1, 1, "pos"),
    ("The person in the leftmost", "x", 1, 1, "pos"),
    # 序数 + 远近
    ("The second crane farthest from the camera", "y", -1, 2, "depth"),
    ("The deer with antlers that is second closest to the camera", "y", 1, 2, "depth"),
    ("The bear closest to the camera", "y", 1, 1, "depth"),
    # 楼层序数：lowest floor 在图下方 -> y 增大
    ("The closed window on the second lowest floor", "y", -1, 2, "pos"),
    ("The window on the third highest floor", "y", 1, 3, "pos"),
    # 中间 / 居中
    ("A man driving a bumper car in the middle background", "x", 1, None, "middle"),
])
def test_parse_constraint_ex(query, axis, direction, rank, kind):
    cons, spans = parse_constraint_ex(query)
    assert cons is not None, query
    assert cons["axis"] == axis
    assert cons["direction"] == direction
    assert cons["rank"] == rank
    assert cons["kind"] == kind
    assert spans, "有约束时必须给出可抹除的 span"


def test_parse_constraint_ex_none_when_no_constraint():
    cons, spans = parse_constraint_ex("The red car parked on the street")
    assert cons is None and spans == []


def test_parse_constraint_ex_first_aid_not_ordinal():
    """``first aid`` / ``first floor`` 里的 first 不是序数，不应触发约束。"""
    cons, _ = parse_constraint_ex("The first aid kit on the wall")
    assert cons is None


def test_parse_constraint_ex_superset_of_matcher():
    """新解析器必须是旧解析器的超集 —— 否则会静默退化掉已覆盖的 query。"""
    cases = [
        "The second vertical green pole of the climbing structure from the left",
        "The second rectangular box mounted on the back wall from the left side",
        "The second vertical panel of the wooden door assembly counting from the left",
        "The second yellow rectangular box mounted on the wall from the left",
        "The first square structure with white perforated panels from the left",
        "The second white patio umbrella in the outdoor seating area from left to right",
        "The first of three buildings of similar height from left to right",
    ]
    for q in cases:
        old = parse_spatial_constraint(q)
        new, _ = parse_constraint_ex(q)
        assert bool(old) <= bool(new), f"退化: {q!r} old={old} new={new}"


# ------------------------------------------------------------ NP 抽取
@pytest.mark.parametrize("query,np_text", [
    ("The second white patio umbrella in the outdoor seating area "
     "from left to right",
     "white patio umbrella in the outdoor seating area"),
    ("The dark floor tile closest to the camera", "dark floor tile"),
    ("The leftmost roadblock", "roadblock"),
    ("The second red and white bumper car from left to right",
     "red and white bumper car"),
    ("The third black swan from bottom to top", "black swan"),
    ("From left to right, the first red letter L", "red letter L"),
    ("The first of three buildings of similar height from left to right",
     "buildings of similar height"),
    ("The second rectangular wall-mounted box from the left on the back wall",
     "rectangular wall-mounted box on the back wall"),
])
def test_extract_target_np(query, np_text):
    r = extract_target(query)
    assert r is not None, query
    assert r["np"].lower() == np_text.lower(), f"{query!r} -> {r['np']!r}"


def test_extract_target_np_has_no_glued_words():
    """抹除后不得把相邻词粘成伪词（历史 bug：umbrellaoutdoor）。"""
    r = extract_target(
        "The second white patio umbrella in the outdoor seating area from left to right")
    assert "umbrellaoutdoor" not in r["np"].lower()


def test_extract_target_returns_none_for_unusable():
    """纯标点/单字停用词不构成可用 NP。"""
    assert extract_target("the") is None
    assert extract_target("   ") is None
    assert extract_target("") is None


def test_extract_target_pure_appearance_keeps_constraint_none():
    r = extract_target("The white car parked beside the curb")
    assert r is not None and r["constraint"] is None


def test_extract_target_same_np_groups_ordinal_siblings():
    """同图的序数兄弟 query 必须抽到同一个 NP —— 这是枚举调用可复用的前提。"""
    qs = [
        "The second black swan from bottom to top",
        "The third black swan from bottom to top",
        "The first black swan from bottom to top",
    ]
    nps = {extract_target(q)["np"].lower() for q in qs}
    assert nps == {"black swan"}


def test_is_valid_np_rejects_overlong():
    assert not is_valid_np(" ".join(["word"] * 20))
    assert is_valid_np("red car")


# ------------------------------------------------------------ 确定性选择
def test_box_centers():
    assert box_centers([[0.0, 0.0, 0.4, 0.4], [0.2, 0.0, 0.6, 0.4]], "x") == \
        pytest.approx([0.2, 0.4])
    assert box_centers([[0.0, 0.0, 0.4, 0.4], [0.0, 0.2, 0.4, 0.6]], "y") == \
        pytest.approx([0.2, 0.4])


def _x_cons(rank, direction=1):
    return {"axis": "x", "direction": direction, "rank": rank, "kind": "pos"}


def test_select_index_left_to_right():
    vals = [0.8, 0.1, 0.5, 0.3]
    assert select_index(vals, _x_cons(1)) == 1   # 最左 = 最小
    assert select_index(vals, _x_cons(2)) == 3
    assert select_index(vals, _x_cons(4)) == 0


def test_select_index_right_to_left():
    vals = [0.8, 0.1, 0.5, 0.3]
    assert select_index(vals, _x_cons(1, direction=-1)) == 0  # 最右 = 最大
    assert select_index(vals, _x_cons(2, direction=-1)) == 2


def test_select_index_clamps_out_of_range_rank():
    vals = [0.3, 0.7]
    assert select_index(vals, _x_cons(9)) == 1
    assert select_index(vals, _x_cons(0)) == 0


def test_select_index_middle():
    vals = [0.9, 0.1, 0.5]
    cons = {"axis": "x", "direction": 1, "rank": None, "kind": "middle"}
    assert select_index(vals, cons) == 2


def test_select_index_depth_uses_smallest_as_nearest():
    depth = [5.0, 1.0, 3.0]          # 越小越近
    assert select_index(depth, {"axis": "y", "direction": 1, "rank": 1,
                                "kind": "depth"}) == 1
    assert select_index(depth, {"axis": "y", "direction": -1, "rank": 2,
                                "kind": "depth"}) == 2


def test_select_index_empty_or_no_constraint():
    assert select_index([], _x_cons(1)) is None
    assert select_index([0.1], None) is None


def test_select_box_pos_axis():
    boxes = [[0.7, 0.0, 0.9, 0.2], [0.1, 0.0, 0.3, 0.2]]
    box, idx = select_box(boxes, _x_cons(1))
    assert idx == 1 and box == boxes[1]


def test_select_box_depth_requires_depth_means():
    """深度序缺深度图时必须回退，不能拿 y 中心冒充远近。"""
    boxes = [[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.7, 0.7]]
    cons = {"axis": "y", "direction": 1, "rank": 1, "kind": "depth"}
    assert select_box(boxes, cons, None) == (None, None)
    box, idx = select_box(boxes, cons, [9.0, 2.0])
    assert idx == 1 and box == boxes[1]


def test_select_box_no_constraint_returns_none():
    """无约束不得静默取第一个框 —— 交给上游原有路径。"""
    assert select_box([[0.0, 0.0, 0.2, 0.2]], None) == (None, None)
    assert select_box([], _x_cons(1)) == (None, None)