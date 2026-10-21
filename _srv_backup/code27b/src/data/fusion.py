"""多模态模态适配与融合模块。

对应《零样本视觉定位模型选型建议（最终版）》§二「推荐技术路线」：

    - 阶段二（ModPrompt）：红外/深度 → 伪 RGB，接入冻结的 RGB 定位基座
      （Grounding DINO），保留零样本能力。ModPrompt 原文用可训练 U-Net 适配器；
      本工程零样本、不训练，故以**训练无关的确定性映射**近似。
    - 阶段三（GatedFusion-Net）：逐像素 Sigmoid 门控融合；文档指出其适配本数据集
      深度无效区占比高（中位 25.18%）的问题——无效深度像素门控置 0。
    - 阶段五（CityViMD-Net）：深度按 ~20000mm 截断归一化（见 ``preprocess.DEPTH_VALID_MAX``）。

所有函数接收/返回 numpy 数组，输入三模态空间对齐（H、W 一致）。
"""
from __future__ import annotations

import numpy as np

from src.data.preprocess import denoise_infrared, normalize_rgb

_EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数值稳定的 Sigmoid（门控激活）。"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def infrared_pseudo_rgb(ir: np.ndarray, ksize: int = 3) -> np.ndarray:
    """红外 → 伪 RGB（阶段二 模态适配）。

    红外本质为灰度三通道堆叠，去噪后取灰度并复制为 3 通道。

    Args:
        ir: uint8 (H, W, 3) 或 (H, W) 红外图。
        ksize: 中值滤波核大小。

    Returns:
        float32 (H, W, 3)，取值 [0, 1]。
    """
    gray01 = denoise_infrared(ir, ksize) / 255.0  # (H, W, 3) float [0,1]
    return np.clip(gray01, 0.0, 1.0).astype(np.float32)


def depth_pseudo_rgb(depth_norm: np.ndarray,
                     valid_mask: np.ndarray | None = None) -> np.ndarray:
    """深度 → 伪 RGB（阶段二 模态适配）。

    归一化深度复制为 3 通道；无效区（``valid_mask=False``）置 0，避免用无效
    深度误导冻结检测器。

    Args:
        depth_norm: [0,1] 归一化深度，(H, W) 或 (H, W, 1)。
        valid_mask: bool (H, W) 深度有效掩码；None 视为全部有效。

    Returns:
        float32 (H, W, 3)，取值 [0, 1]。
    """
    d = depth_norm.astype(np.float32)
    if d.ndim == 2:
        d = d[..., None]
    if valid_mask is not None:
        m = valid_mask.astype(np.float32)
        if m.ndim == 2:
            m = m[..., None]
        d = d * m
    d = np.clip(d, 0.0, 1.0)
    return np.repeat(d, 3, axis=2).astype(np.float32)


def gated_fusion(rgb: np.ndarray,
                 ir: np.ndarray,
                 depth_norm: np.ndarray | None = None,
                 valid_mask: np.ndarray | None = None,
                 *,
                 low_light_threshold: float = 60.0,
                 gate_scale: float = 30.0,
                 ir_max_weight: float = 1.0,
                 depth_max_weight: float = 1.0) -> np.ndarray:
    """逐像素 Sigmoid 门控的三模态融合 → 冻结 Grounding DINO 的伪 RGB 输入。

    门控设计（训练无关，依据文档 §二）::

        gate_rgb   = 1
        gate_ir    = sigmoid((low_light_threshold - 逐像素亮度) / gate_scale)
        gate_depth = valid_mask                     # 无效深度硬置 0

        fused = (rgb + ir_max_weight·gate_ir·ir + depth_max_weight·gate_depth·depth)
                / (1 + ir_max_weight·gate_ir + depth_max_weight·gate_depth)

    该式把现有"全局低光开关 + 固定 alpha"细化为**逐像素**门控，并对高无效率的
    深度做掩码处理，几何结构与可见光一致（bbox 仍在可见光坐标系）。

    Args:
        rgb: uint8 (H, W, 3) 可见光。
        ir: uint8 (H, W, 3) 红外。
        depth_norm: [0,1] 归一化深度；None 则融合不含深度。
        valid_mask: bool (H, W) 深度有效掩码；None 视为全部有效。
        low_light_threshold: 低光阈值（与 config 对齐）。
        gate_scale: Sigmoid 坡度（越大过渡越平缓）。
        ir_max_weight / depth_max_weight: 红外/深度门控上限权重。

    Returns:
        uint8 (H, W, 3) 融合图。
    """
    rgb_f = normalize_rgb(rgb)
    ir_rgb = infrared_pseudo_rgb(ir)

    # 逐像素亮度（0~255）→ 逐像素红外门控
    brightness = np.mean(rgb.astype(np.float32), axis=2)
    gate_ir = _sigmoid((low_light_threshold - brightness) / max(gate_scale, _EPS))
    gate_ir = gate_ir[..., None]

    w_ir = float(ir_max_weight) * gate_ir
    num = rgb_f + w_ir * ir_rgb
    den = 1.0 + w_ir

    if depth_norm is not None:
        depth_rgb = depth_pseudo_rgb(depth_norm, valid_mask)
        if valid_mask is not None:
            gate_d = valid_mask.astype(np.float32)
            if gate_d.ndim == 2:
                gate_d = gate_d[..., None]
        else:
            gate_d = np.ones((rgb_f.shape[0], rgb_f.shape[1], 1), dtype=np.float32)
        w_d = float(depth_max_weight) * gate_d
        num = num + w_d * depth_rgb
        den = den + w_d

    fused = np.clip(num / (den + _EPS), 0.0, 1.0)
    return (fused * 255.0).astype(np.uint8)


def build_split_modalities(rgb: np.ndarray,
                           ir: np.ndarray,
                           depth_norm: np.ndarray | None = None,
                           valid_mask: np.ndarray | None = None) -> dict:
    """构造「分模态图像」：红外、深度各渲染为 uint8 三通道，供多图输入的 VLM
    （Qwen3-VL）分别摄入，替代「伪 RGB 融合」单图路径。可见光不在此返回（原样使用）。

    Args:
        rgb: uint8 (H,W,3) 可见光（仅用于对齐，不参与返回）。
        ir: uint8 (H,W,3) 红外。
        depth_norm: [0,1] 归一化深度；None 则不含 depth。
        valid_mask: bool (H,W) 深度有效掩码。

    Returns:
        ``{"infrared": uint8(H,W,3), "depth": uint8(H,W,3)?}``（无 depth 输入时不含 depth 键）。
    """
    out = {
        "infrared": np.clip(infrared_pseudo_rgb(ir) * 255.0, 0, 255).astype(np.uint8),
    }
    if depth_norm is not None:
        out["depth"] = np.clip(
            depth_pseudo_rgb(depth_norm, valid_mask) * 255.0, 0, 255
        ).astype(np.uint8)
    return out
