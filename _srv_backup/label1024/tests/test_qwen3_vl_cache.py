"""单元测试：Qwen3-VL 视觉特征缓存装配（用 stub 内层模型，不加载权重）。

依赖 torch（``qwen3_vl`` 经 ``BaseGroundingModel`` 引入）；无 torch 环境自动跳过。
运行：python -m pytest tests/test_qwen3_vl_cache.py -q
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import pytest

torch = pytest.importorskip("torch")

from src.models.qwen3_vl import Qwen3VLModel
from src.utils.cache import LRUCache


class _StubInner:
    """模拟内层模型的 get_image_features（计数调用）。"""

    def __init__(self):
        self.calls = 0

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        self.calls += 1
        return {"n": self.calls, "sum": float(pixel_values.sum())}


def _wrap(inner):
    """构造带最小属性的 dummy self 并安装缓存包装。"""
    dummy = SimpleNamespace(
        model=SimpleNamespace(model=inner),
        _vision_cache=LRUCache(4),
        _active_image_key=None,
        feature_cache=True,
    )
    Qwen3VLModel._install_vision_cache(dummy)
    return dummy


def test_vision_cache_reuses_within_same_image():
    inner = _StubInner()
    dummy = _wrap(inner)
    pv = torch.zeros(1, 3, 2, 2)
    dummy._active_image_key = "A"
    out1 = inner.get_image_features(pv, image_grid_thw=None)
    out2 = inner.get_image_features(pv, image_grid_thw=None)
    assert inner.calls == 1            # 视觉塔只算一次
    assert out1 is out2                # 复用同一输出对象


def test_vision_cache_recomputes_for_new_image():
    inner = _StubInner()
    dummy = _wrap(inner)
    pv = torch.zeros(1, 3, 2, 2)
    dummy._active_image_key = "A"
    inner.get_image_features(pv, image_grid_thw=None)
    dummy._active_image_key = "B"
    inner.get_image_features(pv, image_grid_thw=None)
    assert inner.calls == 2            # 换图重新计算


def test_vision_cache_no_key_still_calls_original():
    inner = _StubInner()
    dummy = _wrap(inner)
    dummy._active_image_key = None
    inner.get_image_features(torch.zeros(1, 3, 2, 2), image_grid_thw=None)
    assert inner.calls == 1            # 无键 → 直接走原实现


def test_vision_cache_disabled_when_missing():
    dummy = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace()),  # 无 get_image_features
        _vision_cache=LRUCache(2),
        _active_image_key=None,
        feature_cache=True,
    )
    Qwen3VLModel._install_vision_cache(dummy)
    assert dummy.feature_cache is False
