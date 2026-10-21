# -*- coding: utf-8 -*-
"""序数链（跨查询一致性约束）的提取与单调性修正测试。"""
import pytest

from src.inference.chains import (
    chain_of,
    classify_chain,
    enforce_ordinal_chains,
    pava_non_decreasing,
    residual_tokens,
)

L, R = 1, -1  # direction：+1 = rank 越大坐标越大（from the left / top）


def cons(axis, direction, rank):
    return {"axis": axis, "direction": direction, "rank": rank}


def cx(b):
    return (b[0] + b[2]) / 2


def cy(b):
    return (b[1] + b[3]) / 2


def mk(c, y=0.1, w=0.04, h=0.10):
    return [c - w / 2, y, c + w / 2, y + h]


class TestResidualTokens:
    def test_drops_ordinal_and_axis_words(self):
        assert residual_tokens("The first tree trunk from the left") == \
            frozenset({"tree", "trunk"})

    def test_drops_stopwords(self):
        assert residual_tokens("a red car on the road") == \
            frozenset({"red", "car", "road"})

    def test_case_and_punctuation(self):
        assert residual_tokens("The Second  Car, From The Right!") == frozenset({"car"})

    def test_superlative_dropped(self):
        assert residual_tokens("the leftmost pedestrian") == frozenset({"pedestrian"})


class TestChainOf:
    def test_from_left(self):
        ch = chain_of("The first tree trunk from the left")
        assert (ch["axis"], ch["direction"], ch["rank"]) == ("x", L, 1)
        assert ch["key"] == frozenset({"tree", "trunk"})

    def test_from_right_is_negative(self):
        ch = chain_of("the third car from the right")
        assert (ch["axis"], ch["direction"], ch["rank"]) == ("x", R, 3)

    def test_vertical_axis(self):
        ch = chain_of("the second window from the bottom")
        assert (ch["axis"], ch["direction"], ch["rank"]) == ("y", R, 2)

    def test_non_chain_returns_none(self):
        assert chain_of("a red car") is None

    def test_depth_kind_returns_none(self):
        assert chain_of("the car closest to the camera") is None
        assert chain_of("the farthest building") is None


class TestClassifyChain:
    def test_ordered(self):
        items = [("q1", "", mk(0.1), cons("x", L, 1)),
                 ("q2", "", mk(0.4), cons("x", L, 2)),
                 ("q3", "", mk(0.7), cons("x", L, 3))]
        assert classify_chain(items) == "ordered"

    def test_broken_without_explanation_is_A(self):
        items = [("q1", "", mk(0.6), cons("x", L, 1)),
                 ("q2", "", mk(0.2), cons("x", L, 2))]
        assert classify_chain(items) == "A"

    def test_duplicate_boxes_is_B_dup(self):
        box = mk(0.15)
        items = [("q1", "", box, cons("x", L, 1)),
                 ("q2", "", list(box), cons("x", L, 2)),
                 ("q3", "", mk(0.55), cons("x", L, 3))]
        assert classify_chain(items) == "B_dup"

    def test_tie_is_B_tie(self):
        items = [("q1", "", mk(0.3, y=0.1), cons("x", L, 1)),
                 ("q2", "", mk(0.3, y=0.5), cons("x", L, 2)),
                 ("q3", "", mk(0.9), cons("x", L, 3))]
        assert classify_chain(items) == "B_tie"

    def test_gap_is_B_broken(self):
        items = [("q1", "", mk(0.10), cons("x", L, 1)),
                 ("q2", "", mk(0.05), cons("x", L, 2)),
                 ("q3", "", mk(0.15), cons("x", L, 3)),
                 ("q4", "", mk(0.90), cons("x", L, 4))]
        assert classify_chain(items) == "B_broken"

    def test_mixed_axis_is_skipped(self):
        items = [("q1", "", mk(0.1), cons("x", L, 1)),
                 ("q2", "", mk(0.4), cons("y", L, 2))]
        assert classify_chain(items) == "skip_axis"

    def test_missing_rank_is_skipped(self):
        items = [("q1", "", mk(0.1), cons("x", L, 1)),
                 ("q2", "", mk(0.4), cons("x", L, 3))]
        assert classify_chain(items) == "skip_rank"


class TestPava:
    def test_monotone_unchanged(self):
        assert pava_non_decreasing([0.1, 0.2, 0.3]) == pytest.approx([0.1, 0.2, 0.3])

    def test_pool_adjacent_violators(self):
        assert pava_non_decreasing([0.6, 0.2]) == pytest.approx([0.4, 0.4])

    def test_pool_block(self):
        # [0.5, 0.2, 0.3] 一路向左合并为 1/3（L2 最优）
        assert pava_non_decreasing([0.1, 0.5, 0.2, 0.3]) == \
            pytest.approx([0.1, 1 / 3, 1 / 3, 1 / 3])

    def test_single_value(self):
        assert pava_non_decreasing([0.7]) == pytest.approx([0.7])


class TestEnforceOrdinalChains:
    Q = ["The first tree trunk from the left",
         "The second tree trunk from the left",
         "The third tree trunk from the left"]

    def test_broken_chain_is_reassigned(self):
        boxes = [[0.10, 0.1, 0.20, 0.9], [0.40, 0.1, 0.50, 0.9],
                 [0.05, 0.1, 0.15, 0.9]]
        out = enforce_ordinal_chains(self.Q, boxes)
        assert [cx(b) for b in out] == pytest.approx([0.10, 0.15, 0.45])

    def test_ordered_chain_is_idempotent(self):
        boxes = [[0.10, 0.1, 0.20, 0.9], [0.40, 0.1, 0.50, 0.9],
                 [0.70, 0.1, 0.80, 0.9]]
        assert enforce_ordinal_chains(self.Q, boxes) == boxes

    def test_none_and_non_chain_untouched(self):
        boxes = [None, [0.5, 0.5, 0.6, 0.6], None]
        assert enforce_ordinal_chains(["a car", "a tree", "a bus"], boxes) == \
            [None, [0.5, 0.5, 0.6, 0.6], None]

    def test_single_member_chain_untouched(self):
        box = [[0.9, 0.1, 1.0, 0.2]]
        assert enforce_ordinal_chains(["the first car from the left"], box) == box

    def test_different_objects_not_mixed(self):
        q = ["The first car from the left", "The first tree from the left",
             "The second car from the left", "The second tree from the left"]
        boxes = [[0.2, 0.1, 0.3, 0.2], [0.9, 0.1, 1.0, 0.2],
                 [0.4, 0.1, 0.5, 0.2], [0.1, 0.1, 0.2, 0.2]]
        out = enforce_ordinal_chains(q, boxes)
        assert [cx(out[0]), cx(out[2])] == pytest.approx([0.25, 0.45])   # 车链已单调
        assert [cx(out[1]), cx(out[3])] == pytest.approx([0.15, 0.95])   # 树链被交换

    def test_from_right_direction(self):
        q = ["The first car from the right", "The second car from the right"]
        boxes = [[0.2, 0.1, 0.3, 0.2], [0.8, 0.1, 0.9, 0.2]]
        out = enforce_ordinal_chains(q, boxes)
        assert [cx(b) for b in out] == pytest.approx([0.85, 0.25])

    def test_project_mode_uses_pava(self):
        q = ["The first car from the left", "The second car from the left"]
        boxes = [[0.55, 0.1, 0.65, 0.2], [0.15, 0.1, 0.25, 0.2]]
        out = enforce_ordinal_chains(q, boxes, mode="project")
        assert [cx(b) for b in out] == pytest.approx([0.4, 0.4])

    def test_gap_ratio_pushes_ties_apart(self):
        q = ["The first car from the left", "The second car from the left"]
        boxes = [[0.25, 0.1, 0.35, 0.2], [0.25, 0.5, 0.35, 0.6]]
        out = enforce_ordinal_chains(q, boxes, gap_ratio=1.0)
        assert [cx(b) for b in out] == pytest.approx([0.3, 0.4])

    def test_max_shift_caps_movement(self):
        boxes = [[0.10, 0.1, 0.20, 0.9], [0.40, 0.1, 0.50, 0.9],
                 [0.05, 0.1, 0.15, 0.9]]
        out = enforce_ordinal_chains(self.Q, boxes, max_shift=0.05)
        for b0, b1 in zip(boxes, out):
            assert abs(cx(b1) - cx(b0)) <= 0.05 + 1e-9

    def test_y_axis_chain(self):
        q = ["the first window from the top", "the second window from the top"]
        boxes = [[0.1, 0.6, 0.2, 0.7], [0.1, 0.2, 0.2, 0.3]]
        out = enforce_ordinal_chains(q, boxes)
        assert [cy(b) for b in out] == pytest.approx([0.25, 0.65])

    def test_box_size_preserved_and_clipped(self):
        q = ["The first car from the left", "The second car from the left"]
        boxes = [[0.90, 0.1, 1.00, 0.2], [0.05, 0.1, 0.15, 0.2]]
        out = enforce_ordinal_chains(q, boxes)
        for b in out:
            assert 0.0 <= b[0] <= b[2] <= 1.0
            assert b[2] - b[0] == pytest.approx(0.10)

    def test_output_length_matches_input(self):
        out = enforce_ordinal_chains(self.Q, [None] * 3)
        assert len(out) == 3