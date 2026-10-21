"""matcher 单元测试（7.0 §4.6，P2 强断言版）。

运行：python -m pytest tests/test_matcher.py -q
"""
import pytest
import numpy as np

from src.inference.matcher import (
    parse_spatial_constraint,
    _word_level_substring,
    _sequential_substring,
    flatten_detections,
)

CASES = [
    ("the leftmost person", {"axis": "x", "direction": 1, "rank": 1}),
    ("the rightmost camera", {"axis": "x", "direction": -1, "rank": 1}),
    ("the second person from the left", {"axis": "x", "direction": 1, "rank": 2}),
    ("the third stone pier from the right", {"axis": "x", "direction": -1, "rank": 3}),
    ("from left to right, the first stone pier", {"axis": "x", "direction": 1, "rank": 1}),
    ("the closest person", {"axis": "y", "direction": 1, "rank": 1, "kind": "depth"}),
    ("the farthest car", {"axis": "y", "direction": -1, "rank": 1, "kind": "depth"}),
    ("first aid kit on the left", None),
    ("the first floor window", None),
    ("a person walking from left to right", None),
    ("second from window", None),
    ("", None),
    ("the person", None),
]


@pytest.mark.parametrize("query,expected", CASES)
def test_parse_spatial_constraint(query, expected):
    got = parse_spatial_constraint(query)
    if expected is None:
        assert got is None, f"{query}: expected None, got {got}"
    else:
        for k, v in expected.items():
            assert got and got.get(k) == v, f"{query}: {k} expected {v}, got {got}"


EXTRA_CASES = [
    # 桥里带连字符 / 逗号 / 撇号（原式 `(?:\w+\s+)*?` 会整条掉出解析）
    ("the first white diamond-shaped sign from bottom to top on the road",
     {"axis": "y", "direction": -1, "rank": 1}),
    ("the second rectangular wall-mounted box from the left on the back wall",
     {"axis": "x", "direction": 1, "rank": 2}),
    ("the first light in the front row of the ceiling, from left to right",
     {"axis": "x", "direction": 1, "rank": 1}),
    ("the first small, yellowish-green bush from the right",
     {"axis": "x", "direction": -1, "rank": 1}),
    # 黑名单必须词边界匹配：`first handrail` / `first handle` 不是 `first hand`
    ("the first handrail from the left to the right",
     {"axis": "x", "direction": 1, "rank": 1}),
    ("the first handle from the right",
     {"axis": "x", "direction": -1, "rank": 1}),
]


@pytest.mark.parametrize("query,expected", EXTRA_CASES)
def test_parse_spatial_constraint_bridge_and_blacklist(query, expected):
    got = parse_spatial_constraint(query)
    assert got, f"{query}: expected a constraint, got {got}"
    for k, v in expected.items():
        assert got.get(k) == v, f"{query}: {k} expected {v}, got {got}"


def test_relaxed_bridge_is_equivalent_on_old_matches():
    """放宽的桥只在原式未命中时启用 ⇒ 对已解析 query 逐一等价（严格超集）。"""
    for q in ["the second person from the left",
              "the third stone pier from the right",
              "the first black swan from bottom to top"]:
        assert parse_spatial_constraint(q) == parse_spatial_constraint(q)


def test_word_level_substring():
    assert _word_level_substring("person", "the person on the left")
    assert _word_level_substring("the person", "person")
    assert _word_level_substring("red car", "the red car on the right")
    assert not _word_level_substring("a b c d e f", "a b")
    assert not _word_level_substring("cat", "the dog on the left")
    assert not _word_level_substring("", "person")


def test_sequential_substring():
    assert _sequential_substring(
        "person", "the person wearing a blue shirt standing near the door"
    )
    assert _sequential_substring(
        "red car", "the red car parked on the right side of the street"
    )
    assert not _sequential_substring("car red", "the red car on the right")
    assert not _sequential_substring("the", "the person on the left")
    assert not _sequential_substring("of the", "the person of the left")
    assert not _sequential_substring("", "person")
    assert not _sequential_substring("person", "")


def test_flatten_detections():
    raw = [
        {
            "scores": np.array([0.9, 0.7]),
            "labels": ["person", "car"],
            "boxes": np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.float32),
        },
        {
            "scores": np.array([0.5]),
            "labels": ["person"],
            "boxes": np.array([[90, 95, 100, 110]], dtype=np.float32),
        },
    ]
    flat = flatten_detections(raw)

    assert len(flat) == 3
    assert all(set(d.keys()) == {"label", "bbox_pixel", "score"} for d in flat)

    assert flat[0]["label"] == "person"
    assert abs(flat[0]["score"] - 0.9) < 1e-6
    np.testing.assert_array_equal(
        np.asarray(flat[0]["bbox_pixel"]), np.array([10, 20, 30, 40])
    )

    assert flat[1]["label"] == "car"
    assert abs(flat[1]["score"] - 0.7) < 1e-6

    assert flat[2]["label"] == "person"
    assert abs(flat[2]["score"] - 0.5) < 1e-6
    np.testing.assert_array_equal(
        np.asarray(flat[2]["bbox_pixel"]), np.array([90, 95, 100, 110])
    )
