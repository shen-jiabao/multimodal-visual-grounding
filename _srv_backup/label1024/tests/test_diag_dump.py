# -*- coding: utf-8 -*-
"""诊断落盘采集的单测（不依赖 torch / 模型权重）。

覆盖 `src/models/qwen3_vl.py` 的 `_predict_multi` / `_predict_single` 诊断分支：
  - 打开 diag 后：解析出的 (label, box) 全部入 `pairs`（**含匹配失败的**）、
    匹配失败的 label 入 `unmatched`、非法框计入 `dropped_invalid`；
  - 关闭 diag（桩件无 `_collect_diag`）时完全静默，且返回的框与打开时一致
    （这是"纯观测、不改预测"的核心保证）。

运行：python -m pytest tests/test_diag_dump.py -q
"""
from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.models.qwen3_vl import Qwen3VLModel


class StubModel:
    """只实现 `_predict_multi` / `_predict_single` 依赖的最小接口。"""

    # 复用真实实现（未绑定函数 → 绑定到本桩件）
    _accept = Qwen3VLModel._accept
    _unit = Qwen3VLModel._unit
    _vote = Qwen3VLModel._vote
    _predict_multi = Qwen3VLModel._predict_multi
    _predict_single = Qwen3VLModel._predict_single
    _diag_on = Qwen3VLModel._diag_on
    reset_diag = Qwen3VLModel.reset_diag

    def __init__(self, replies, collect_diag=True):
        self._replies = list(replies)
        self._collect_diag = collect_diag
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.default_score = 1.0
        self.num_samples = len(self._replies)
        self.vote_iou_threshold = 0.5
        self.vote_min_votes = 1
        self.reset_diag()

    def _generate(self, images, prompt, num_return_sequences=1):
        return list(self._replies)


W, H = 200, 100
MULTI_IMAGES = ["visible RGB image", "infrared thermal image", "depth map"]


def reply_with(*pairs):
    """构造模型回复：[{"label": ..., "bbox_2d": [...]}, ...]"""
    import json
    return json.dumps([{"label": lb, "bbox_2d": list(b)} for lb, b in pairs])


class TestMultiBoxDiag:
    def test_unmatched_label_is_recorded(self):
        m = StubModel([reply_with(
            ("a red car", [0, 0, 500, 500]),
            ("banana", [100, 100, 200, 200]),      # 与任何 query 都不匹配
        )], collect_diag=True)
        labels, boxes, _ = m._predict_multi(MULTI_IMAGES, [], ["a red car"], W, H)

        assert labels == ["a red car"]
        assert len(boxes) == 1
        # 全部 pair 入档（含匹配失败的），这是区分"截断 vs 回显失败"的关键证据
        assert len(m._diag["pairs"]) == 2
        assert m._diag["unmatched"] == ["banana"]
        assert m._diag["matched"] == 1
        assert m._diag["pairs"][1]["matched_query"] is None
        assert m._diag["pairs"][1]["label"] == "banana"

    def test_truncated_reply_keeps_partial_pairs(self):
        # 模拟 512 token 截断：JSON 未闭合，`_RE_JSON_OBJ` 仍能抓到完整的几个对象
        truncated = ('[{"label": "a red car", "bbox_2d": [0, 0, 500, 500]}, '
                     '{"label": "a dog", "bbox_2d": [600, 600, 900, 900]}, '
                     '{"label": "a bike", "bbox_2d": [10')
        m = StubModel([truncated], collect_diag=True)
        labels, boxes, _ = m._predict_multi(
            MULTI_IMAGES, [], ["a red car", "a dog", "a bike"], W, H)

        assert labels == ["a red car", "a dog"]      # 第三条无框
        assert len(boxes) == 2
        assert m._diag["replies"] == [truncated]
        # 未闭合的 "{" 数量 > 0 —— 截断的直接证据
        assert truncated.count("{") - truncated.count("}") == 1

    def test_invalid_box_counted_as_dropped(self):
        # x2 < x1 且越界：coords_1000_to_unit 会交换，但 2000 越界 → 非法
        m = StubModel([reply_with(
            ("a red car", [0, 0, 500, 500]),
            ("a dog", [0, 0, 2000, 500]),
        )], collect_diag=True)
        labels, _, _ = m._predict_multi(MULTI_IMAGES, [], ["a red car", "a dog"], W, H)
        assert labels == ["a red car"]
        assert m._diag["dropped_invalid"] == 1
        assert m._diag["matched"] == 1

    def test_diag_off_is_silent_but_same_boxes(self):
        rep = [reply_with(("a red car", [0, 0, 500, 500]),
                          ("banana", [10, 10, 20, 20]))]
        on = StubModel(rep, collect_diag=True)
        off = StubModel(rep, collect_diag=False)

        r_on = on._predict_multi(MULTI_IMAGES, [], ["a red car"], W, H)
        r_off = off._predict_multi(MULTI_IMAGES, [], ["a red car"], W, H)

        assert r_on == r_off                     # 纯观测：不改预测
        assert len(on._diag["pairs"]) == 2
        assert off._diag["pairs"] == []           # 关闭时完全不采集

    def test_diag_off_without_attribute_never_raises(self):
        """桩件完全没有 _collect_diag 时（旧单测的 SimpleNamespace 形态）不报错。"""
        m = StubModel([reply_with(("a red car", [0, 0, 500, 500]))], collect_diag=True)
        del m._collect_diag
        labels, _, _ = m._predict_multi(MULTI_IMAGES, [], ["a red car"], W, H)
        assert labels == ["a red car"]

    def test_multi_sample_records_candidates_and_votes(self):
        rep = [
            reply_with(("a red car", [0, 0, 500, 500])),
            reply_with(("a red car", [5, 5, 505, 505])),   # IoU 高 → 投票聚合
        ]
        m = StubModel(rep, collect_diag=True)
        labels, boxes, _ = m._predict_multi(MULTI_IMAGES, [], ["a red car"], W, H)
        assert labels == ["a red car"]
        assert len(m._diag["candidates"]["a red car"]) == 2
        v = m._diag["votes"]["a red car"]
        assert v["n_candidates"] == 2
        assert v["voted"] is not None


class TestSingleBoxDiag:
    def test_unmatched_single_still_recorded(self):
        m = StubModel(['{"bbox_2d": [0, 0, 500, 500]}'], collect_diag=True)
        labels, boxes, _ = m._predict_single(MULTI_IMAGES, [], ["a red car"], W, H)
        assert labels == ["a red car"]
        assert len(m._diag["pairs"]) == 1
        assert m._diag["pairs"][0]["matched_query"] == "a red car"

    def test_unparsable_single_records_none_box(self):
        m = StubModel(["I cannot find it."], collect_diag=True)
        labels, boxes, _ = m._predict_single(MULTI_IMAGES, [], ["a red car"], W, H)
        assert labels == [] and boxes == []
        assert m._diag["pairs"][0]["box1000"] is None
        assert m._diag["pairs"][0]["matched_query"] is None


class TestDiagReset:
    def test_reset_diag_clears_previous(self):
        m = StubModel([reply_with(("a red car", [0, 0, 500, 500]))], collect_diag=True)
        m._predict_multi(MULTI_IMAGES, [], ["a red car"], W, H)
        assert m._diag["pairs"]
        m.reset_diag()
        assert m._diag["pairs"] == []
        assert m._diag["unmatched"] == []
        assert m._diag["matched"] == 0

    def test_diag_on_property_defaults_false(self):
        ns = types.SimpleNamespace()
        assert Qwen3VLModel._diag_on.fget(ns) is False
        ns._collect_diag = True
        assert Qwen3VLModel._diag_on.fget(ns) is True
