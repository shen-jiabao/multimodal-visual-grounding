"""单元测试：多模态模态适配与门控融合（选型建议 §二 技术路线，不依赖 torch）。

运行：python -m pytest tests/test_fusion.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from src.data.fusion import (
    build_split_modalities,
    depth_pseudo_rgb,
    gated_fusion,
    infrared_pseudo_rgb,
)
from src.data.preprocess import build_grounding_input


def _solid(h: int, w: int, value: int) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


# ---------- 伪 RGB 映射（阶段二） ----------

def test_infrared_pseudo_rgb_shape_and_range():
    out = infrared_pseudo_rgb(_solid(8, 6, 120))
    assert out.shape == (8, 6, 3)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0
    # 灰度三通道复制：三通道相等
    assert np.allclose(out[..., 0], out[..., 1])
    assert np.allclose(out[..., 1], out[..., 2])


def test_depth_pseudo_rgb_masks_invalid():
    depth = np.full((4, 4), 0.8, dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 0] = False
    out = depth_pseudo_rgb(depth, mask)
    assert out.shape == (4, 4, 3)
    assert out[0, 0].tolist() == [0.0, 0.0, 0.0]      # 无效区置 0
    assert abs(out[1, 1, 0] - 0.8) < 1e-6             # 有效区保留


# ---------- 门控融合（阶段三） ----------

def test_gated_fusion_bright_scene_is_rgb():
    # 亮场景：红外门控 ≈0 → 融合 ≈ 可见光
    rgb = _solid(8, 8, 220)
    out = gated_fusion(rgb, rgb, low_light_threshold=60.0)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
    assert np.max(np.abs(out.astype(np.int16) - rgb.astype(np.int16))) <= 1


def test_gated_fusion_dark_scene_uses_infrared():
    rgb = _solid(8, 8, 20)     # 低光
    ir = _solid(8, 8, 240)     # 红外亮
    out = gated_fusion(rgb, ir, low_light_threshold=60.0)
    assert out.mean() > rgb.mean()          # 红外抬亮
    assert out.mean() <= ir.mean()


def test_gated_fusion_invalid_depth_ignored():
    rgb = _solid(8, 8, 30)
    ir = _solid(8, 8, 200)
    depth = np.full((8, 8), 0.9, dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)     # 全部无效
    assert np.array_equal(
        gated_fusion(rgb, ir, depth, mask),
        gated_fusion(rgb, ir, None, None),  # 无效深度无贡献
    )


def test_gated_fusion_valid_depth_contributes():
    rgb = _solid(8, 8, 30)
    ir = _solid(8, 8, 30)
    depth = np.full((8, 8), 1.0, dtype=np.float32)     # 远 → 伪RGB=1
    mask = np.ones((8, 8), dtype=bool)
    assert gated_fusion(rgb, ir, depth, mask).mean() > gated_fusion(rgb, ir).mean()


# ---------- build_grounding_input 分派 ----------

def test_build_grounding_input_modes():
    rgb = _solid(6, 6, 20)
    ir = _solid(6, 6, 240)
    depth = np.full((6, 6), 0.5, dtype=np.float32)
    mask = np.ones((6, 6), dtype=bool)

    assert build_grounding_input(rgb, ir, fusion_mode="none") is rgb
    assert build_grounding_input(rgb, ir, enable_fusion=False) is rgb

    gated = build_grounding_input(
        rgb, ir, fusion_mode="gated", low_light_threshold=60.0,
        depth_norm=depth, valid_mask=mask,
    )
    assert gated.shape == (6, 6, 3) and gated.dtype == np.uint8
    assert gated.mean() > rgb.mean()     # 低光 + 有效深度 → 抬亮


# ---------- 分模态图像（多图输入） ----------

def test_build_split_modalities():
    rgb = _solid(6, 6, 20)
    ir = _solid(6, 6, 200)
    depth = np.full((6, 6), 0.5, dtype=np.float32)
    mask = np.ones((6, 6), dtype=bool)

    out = build_split_modalities(rgb, ir, depth, mask)
    assert set(out.keys()) == {"infrared", "depth"}
    for v in out.values():
        assert v.shape == (6, 6, 3) and v.dtype == np.uint8

    # 无 depth 输入时不含 depth 键
    assert set(build_split_modalities(rgb, ir).keys()) == {"infrared"}

    # 深度无效区置 0（黑）
    m2 = mask.copy()
    m2[0, 0] = False
    assert build_split_modalities(rgb, ir, depth, m2)["depth"][0, 0].tolist() == [0, 0, 0]
