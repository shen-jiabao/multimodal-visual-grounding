"""单元测试：评测指标（不依赖 torch）。"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np
import pytest

from src.eval.metrics import acc_at_iou, iou, mean_iou, summarize


def test_iou_identical():
    assert abs(iou([0, 0, 1, 1], [0, 0, 1, 1]) - 1.0) < 1e-12


def test_iou_disjoint():
    assert abs(iou([0, 0, 1, 1], [1, 1, 2, 2]) - 0.0) < 1e-12


def test_iou_partial_overlap():
    assert abs(iou([0, 0, 1, 1], [0.5, 0, 1.5, 1]) - 1.0 / 3.0) < 1e-12


def test_iou_reversed_raises():
    with pytest.raises(ValueError):
        iou([1, 0, 0, 1], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        iou([0, 0, 1, 1], [0, 1, 1, 0])


def test_iou_non_finite_raises():
    with pytest.raises(ValueError):
        iou([float("nan"), 0, 1, 1], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        iou([0, 0, 1, 1], [0, 0, float("inf"), 1])


def test_iou_wrong_length_raises():
    with pytest.raises(ValueError):
        iou([0, 0, 1], [0, 0, 1, 1])
    with pytest.raises(ValueError):
        iou([0, 0, 1, 1, 1], [0, 0, 1, 1])


def test_iou_none_is_zero():
    assert iou(None, [0, 0, 1, 1]) == 0.0
    assert iou([0, 0, 1, 1], None) == 0.0


def test_iou_float32_precision():
    a = np.array([0, 0, 1, 1], dtype=np.float32)
    b = np.array([0.5, 0, 1.5, 1], dtype=np.float32)
    assert abs(iou(a, b) - 1.0 / 3.0) < 1e-12


def test_acc_none_treated_as_wrong():
    assert acc_at_iou([None], [[0, 0, 1, 1]]) == 0.0


def test_acc_length_mismatch_raises():
    with pytest.raises(ValueError):
        acc_at_iou([[0, 0, 1, 1]], [])


def test_acc_empty_is_nan():
    assert math.isnan(acc_at_iou([], []))
    assert math.isnan(mean_iou([], []))


def test_acc_threshold_bounds():
    with pytest.raises(ValueError):
        acc_at_iou([[0, 0, 1, 1]], [[0, 0, 1, 1]], 1.5)
    with pytest.raises(ValueError):
        acc_at_iou([[0, 0, 1, 1]], [[0, 0, 1, 1]], -0.1)


def test_acc_boundary_inclusive():
    preds = [[0, 0, 1, 1]]
    assert acc_at_iou(preds, [[0, 0, 1, 1]], 0.5) == 1.0
    assert acc_at_iou(preds, [[0, 0, 2, 1]], 0.5) == 1.0
    assert acc_at_iou(preds, [[0.5, 0, 1.5, 1]], 0.5) == 0.0


def test_mean_iou_none_as_zero():
    assert abs(mean_iou([None], [[0, 0, 1, 1]]) - 0.0) < 1e-12
    assert abs(mean_iou(
        [[0, 0, 1, 1], None],
        [[0, 0, 1, 1], [1, 0, 2, 1]],
    ) - 0.5) < 1e-12


def test_mean_iou_length_mismatch_raises():
    with pytest.raises(ValueError):
        mean_iou([[0, 0, 1, 1]], [])


def test_summarize():
    preds = [[0, 0, 1, 1], None, [0.5, 0, 1.5, 1]]
    gts = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    out = summarize(preds, gts, 0.5)
    assert out["num_total"] == 3
    assert out["num_none"] == 1
    assert out["num_correct"] == 1
    assert abs(out["acc"] - 1.0 / 3.0) < 1e-12
    assert abs(out["mean_iou"] - 4.0 / 9.0) < 1e-12


def test_summarize_empty():
    out = summarize([], [])
    assert math.isnan(out["acc"])
    assert math.isnan(out["mean_iou"])
    assert out["num_total"] == 0
    assert out["num_none"] == 0
    assert out["num_correct"] == 0


def test_summarize_threshold_bounds():
    with pytest.raises(ValueError):
        summarize([[0, 0, 1, 1]], [[0, 0, 1, 1]], 1.1)
