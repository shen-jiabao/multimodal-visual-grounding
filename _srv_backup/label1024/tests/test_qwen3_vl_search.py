# -*- coding: utf-8 -*-
"""搜索式推理在 Qwen3VLModel 内的接线测试（stub ``_generate``，不加载权重）。

覆盖：贪心单次路径行为不变（得分 = default_score）+ 多候选采样路径的投票聚合。
"""
from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pytest

MODEL_MODULE = "src.models.qwen3_vl"


def _load_model_class():
    """导入 Qwen3VLModel；本地未装 torch 时注入最小 stub（本文件不执行任何 torch 运算）。"""
    if MODEL_MODULE in sys.modules:
        return sys.modules[MODEL_MODULE].Qwen3VLModel
    installed = False
    try:
        import torch  # noqa: F401
    except ImportError:
        sys.modules["torch"] = types.ModuleType("torch")
        installed = True
    try:
        import importlib
        module = importlib.import_module(MODEL_MODULE)
    finally:
        if installed:
            sys.modules.pop("torch", None)
    return module.Qwen3VLModel


Qwen3VLModel = _load_model_class()

MODALITIES = ["visible RGB image"]


class _StubGenerate:
    """按调用次数返回预置回复（模拟采样得到多个候选）。

    ``per_call`` 的每个元素 = **一次** _generate 调用的返回值：str 表示单条回复，
    list 表示该次调用返回的多条候选回复。
    """

    def __init__(self, per_call):
        # 每个元素 = 一次调用返回的回复；str 视为「单条回复」，避免被拆成字符
        self.per_call = [[x] if isinstance(x, str) else list(x) for x in per_call]
        self.calls = 0

    def __call__(self, images, prompt, num_return_sequences: int = 1):
        self.calls += 1
        i = min(self.calls - 1, len(self.per_call) - 1)
        return list(self.per_call[i])


def make_model(per_call, num_samples=1, default_score=1.0):
    m = SimpleNamespace(
        num_samples=num_samples,
        default_score=default_score,
        scale_x=1.0,
        scale_y=1.0,
        vote_iou_threshold=0.5,
        vote_min_votes=1,
        sample_temperature=0.7,
        sample_top_p=0.9,
    )
    m._generate = _StubGenerate(per_call)
    for name in ("_unit", "_vote", "_accept", "_predict_single", "_predict_multi"):
        setattr(m, name, types.MethodType(getattr(Qwen3VLModel, name), m))
    return m


def run_multi(m, texts, w=1000, h=1000):
    return m._predict_multi([None], MODALITIES, texts, w, h)


def run_single(m, texts, w=1000, h=1000):
    return m._predict_single([None], MODALITIES, texts, w, h)


# ---------------- 贪心单次：行为与改造前一致 ----------------

def test_multi_greedy_keeps_default_score():
    reply = ('[{"label": "the red car", "bbox_2d": [100, 200, 300, 400]}, '
             '{"label": "the dog", "bbox_2d": [500, 500, 700, 700]}]')
    m = make_model([[reply]], num_samples=1, default_score=0.42)
    labels, boxes, scores = run_multi(m, ["the red car", "the dog"])
    assert labels == ["the red car", "the dog"]
    assert boxes == [[100.0, 200.0, 300.0, 400.0], [500.0, 500.0, 700.0, 700.0]]
    assert scores == [0.42, 0.42]


def test_multi_greedy_skips_unmatched_label():
    reply = '[{"label": "a bicycle", "bbox_2d": [100, 200, 300, 400]}]'
    m = make_model([[reply]], num_samples=1)
    labels, boxes, _ = run_multi(m, ["the red car"])
    assert labels == [] and boxes == []


def test_single_greedy_uses_default_score():
    m = make_model(['{"bbox_2d": [100, 200, 300, 400]}'], num_samples=1, default_score=0.5)
    labels, boxes, scores = run_single(m, ["the red car"])
    assert labels == ["the red car"]
    assert boxes == [[100.0, 200.0, 300.0, 400.0]]
    assert scores == [0.5]


# ---------------- 多候选采样：投票聚合 ----------------

def test_multi_voting_picks_majority_cluster():
    replies = [
        '[{"label": "the red car", "bbox_2d": [100, 200, 300, 400]}]',
        '[{"label": "the red car", "bbox_2d": [110, 200, 310, 400]}]',
        '[{"label": "the red car", "bbox_2d": [700, 600, 800, 900]}]',
    ]
    # multi 模式一次生成返回全部候选，故整体作为「一次调用」
    m = make_model([replies], num_samples=3)
    labels, boxes, scores = run_multi(m, ["the red car"])
    assert labels == ["the red car"]
    assert boxes[0] == pytest.approx([105.0, 200.0, 305.0, 400.0])
    assert scores[0] == pytest.approx(2 / 3)


def test_multi_voting_all_disagree_falls_back_to_first_cluster():
    replies = [
        '[{"label": "the red car", "bbox_2d": [100, 200, 300, 400]}]',
        '[{"label": "the red car", "bbox_2d": [700, 600, 800, 900]}]',
    ]
    m = make_model([replies], num_samples=2)
    _, boxes, scores = run_multi(m, ["the red car"])
    assert boxes[0] == pytest.approx([100.0, 200.0, 300.0, 400.0])
    assert scores[0] == pytest.approx(0.5)


def test_single_voting_aggregates_per_query():
    per_query = [
        ['{"bbox_2d": [100, 200, 300, 400]}',
         '{"bbox_2d": [110, 200, 310, 400]}',
         '{"bbox_2d": [700, 600, 800, 900]}'],
        ['{"bbox_2d": [0, 0, 100, 100]}',
         '{"bbox_2d": [20, 20, 120, 120]}',
         '{"bbox_2d": [0, 0, 100, 100]}'],
    ]
    m = make_model(per_query, num_samples=3)
    labels, boxes, scores = run_single(m, ["the red car", "the dog"])
    assert labels == ["the red car", "the dog"]
    assert boxes[0] == pytest.approx([105.0, 200.0, 305.0, 400.0])
    assert scores[0] == pytest.approx(2 / 3)
    # 第二组：候选 [0,0,100,100] x2 + [20,20,120,120]（与前者 IoU0.47 < 0.5）
    #  最大簇是两票的 [0,0,100,100]，离群候选被丢弃
    assert boxes[1] == pytest.approx([0.0, 0.0, 100.0, 100.0])
    assert scores[1] == pytest.approx(2 / 3)


def test_vote_single_candidate_returns_default_score():
    m = make_model([['{"bbox_2d": [0, 0, 500, 500]}']], num_samples=1, default_score=0.9)
    box, conf = m._vote([[0.1, 0.1, 0.2, 0.2]])
    assert box == [0.1, 0.1, 0.2, 0.2]
    assert conf == pytest.approx(0.9)


def test_vote_invalid_candidate_is_dropped():
    replies = [
        '[{"label": "the red car", "bbox_2d": [100, 200, 300, 400]}]',
        '[{"label": "the red car", "bbox_2d": [2000, 2000, 3000, 3000]}]',
    ]
    m = make_model([replies], num_samples=2)
    labels, boxes, scores = run_multi(m, ["the red car"])
    assert labels == ["the red car"]                 # 越界候选被 _unit 丢弃
    assert boxes[0] == pytest.approx([100.0, 200.0, 300.0, 400.0])
    assert scores[0] == pytest.approx(1.0)           # 只剩一票  default_score


def test_multi_voting_keeps_first_seen_order():
    replies = [
        ('[{"label": "the dog", "bbox_2d": [500, 500, 700, 700]}, '
         '{"label": "the red car", "bbox_2d": [100, 200, 300, 400]}]'),
    ]
    m = make_model(replies, num_samples=2)
    labels, _, _ = run_multi(m, ["the red car", "the dog"])
    assert labels == ["the dog", "the red car"]