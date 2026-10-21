"""多模态预处理模块。

对应赛题“解题思路（一）数据预处理”：
    - RGB 图像归一化；
    - 红外图像噪声抑制；
    - 深度图像离群值剔除；
    - 低光场景下红外增强可见光（多模态协同，提升鲁棒性）。

所有函数均接收/返回 numpy 数组，保持三模态空间对齐。
"""
from __future__ import annotations

import numpy as np

# 深度图有效范围（毫米），超出视为无效/离群
DEPTH_VALID_MIN = 0
DEPTH_VALID_MAX = 19999


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """RGB 归一化到 [0, 1] 的 float32 数组。

    Args:
        rgb: uint8 (H, W, 3) 可见光图像。
    """
    return rgb.astype(np.float32) / 255.0


def denoise_infrared(ir: np.ndarray, ksize: int = 3) -> np.ndarray:
    """红外图像噪声抑制。

    红外三通道视觉特征高度一致，先转灰度，再用中值滤波去除椒盐噪声，
    最后恢复为三通道以便后续融合。
    """
    ir_f = ir.astype(np.float32)
    # 兼容灰度（H, W）与三通道（H, W, 3）红外输入
    if ir_f.ndim == 2:
        gray = ir_f[..., None]
    else:
        gray = np.mean(ir_f, axis=2, keepdims=True)
    gray = median_filter(gray, ksize)      # 返回 (H, W) 二维
    gray = gray[..., None]                 # 恢复 (H, W, 1)
    return np.repeat(gray, 3, axis=2).astype(np.float32)


def clip_depth(depth: np.ndarray) -> np.ndarray:
    """深度图离群值剔除：将无效/离群深度置为 0，并裁剪到有效范围。"""
    d = depth.astype(np.float32)
    d[d < DEPTH_VALID_MIN] = 0.0
    d[d > DEPTH_VALID_MAX] = DEPTH_VALID_MAX
    return d


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    """深度归一化到 [0, 1]，无效区（0）保持为 0。"""
    d = clip_depth(depth)
    out = np.zeros_like(d, dtype=np.float32)
    valid = d > 0
    out[valid] = d[valid] / DEPTH_VALID_MAX
    return out


def is_low_light(rgb: np.ndarray, threshold: float = 60.0) -> bool:
    """判断可见光图像是否处于低光环境（平均亮度低于阈值）。"""
    return float(np.mean(rgb)) < threshold


def fuse_rgb_infrared(rgb: np.ndarray, ir: np.ndarray,
                      low_light_threshold: float = 60.0,
                      alpha: float = 0.55) -> np.ndarray:
    """低光场景下的红外增强可见光融合。

    当可见光过暗时，将红外热辐射灰度图与可见光亮度加权融合，
    利用红外“弥补低光缺陷、突显热源轮廓”的优势，得到更适合
    视觉编码器的输入图（几何结构与可见光一致，bbox 仍定义在可见光坐标系）。

    Args:
        rgb: uint8 (H, W, 3) 可见光。
        ir: uint8 (H, W, 3) 红外。
        low_light_threshold: 低光判定阈值。
        alpha: 可见光保留比例（1-alpha 为红外权重）。

    Returns:
        uint8 (H, W, 3) 增强后的输入图。
    """
    if not is_low_light(rgb, low_light_threshold):
        return rgb

    ir_gray = denoise_infrared(ir) / 255.0  # (H, W, 3) float [0,1]
    rgb_f = normalize_rgb(rgb)

    # 直方图均衡增强可见光对比度
    rgb_f = _hist_equalize(rgb_f)

    fused = alpha * rgb_f + (1.0 - alpha) * ir_gray
    fused = np.clip(fused, 0.0, 1.0)
    return (fused * 255.0).astype(np.uint8)


def build_grounding_input(rgb: np.ndarray, ir: np.ndarray,
                          enable_fusion: bool = True,
                          low_light_threshold: float = 60.0,
                          fusion_mode: str = "low_light",
                          depth_norm: np.ndarray | None = None,
                          valid_mask: np.ndarray | None = None,
                          fusion_ir_weight: float = 1.0,
                          fusion_depth_weight: float = 1.0,
                          fusion_gate_scale: float = 30.0) -> np.ndarray:
    """构造送入视觉编码器的输入 RGB 图像。

    汇总多模态预处理，按 ``fusion_mode`` 分派：

        - ``"none"``：直接使用可见光；
        - ``"low_light"``（默认）：低光时红外增强（现有基线，行为不变）；
        - ``"gated"``：逐像素门控三模态融合（模态适配 + 门控融合），
          需同时传入 ``depth_norm`` / ``valid_mask``。

    ``enable_fusion=False`` 时一律返回原始可见光。
    """
    if not enable_fusion or fusion_mode == "none":
        return rgb

    if fusion_mode == "gated":
        # 延迟导入：fusion 模块依赖本模块的 denoise_infrared/normalize_rgb，
        # 避免模块级循环导入。
        from src.data.fusion import gated_fusion
        return gated_fusion(
            rgb, ir, depth_norm, valid_mask,
            low_light_threshold=low_light_threshold,
            gate_scale=fusion_gate_scale,
            ir_max_weight=fusion_ir_weight,
            depth_max_weight=fusion_depth_weight,
        )

    return fuse_rgb_infrared(rgb, ir, low_light_threshold=low_light_threshold)


def median_filter(arr: np.ndarray, ksize: int = 3) -> np.ndarray:
    """中值滤波（向量化实现，避免逐像素 Python 循环）。

    优先使用 OpenCV（毫秒级）；无 cv2 时退化为 numpy 滑动窗口向量化。
    """
    if ksize % 2 == 0:
        ksize += 1
    h, w = arr.shape[:2]
    src = arr.reshape(h, w)
    try:
        import cv2
        return cv2.medianBlur(src.astype(np.uint8), ksize)
    except Exception:
        pass
    pad = ksize // 2
    padded = np.pad(src, pad, mode="edge")
    stack = np.stack([padded[i:i + h, j:j + w] for i in range(ksize) for j in range(ksize)], axis=-1)
    return np.median(stack, axis=-1)


def _hist_equalize(img: np.ndarray) -> np.ndarray:
    """对 float [0,1] 图像做近似直方图均衡（提升低光对比度）。"""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[..., c]
        # 用 256 级累积分布做映射
        hist, _ = np.histogram(ch, bins=256, range=(0.0, 1.0))
        cdf = np.cumsum(hist) / (ch.size + 1e-8)
        ch_idx = np.clip((ch * 255).astype(np.int32), 0, 255)
        out[..., c] = cdf[ch_idx]
    return np.clip(out, 0.0, 1.0)


# ---------- 深度置信度分级 ----------
def depth_confidence(valid_ratio: float) -> str:
    if valid_ratio >= 0.75:
        return "high"
    if valid_ratio >= 0.50:
        return "medium"
    if valid_ratio >= 0.10:
        return "low"
    return "skip"


def prepare_depth(sample: dict) -> dict:
    """
    PNG 深度：无效掩码 + 线性归一化 + 四级置信度。
    JPG 深度：灰度归一化，仅视觉先验（confidence="visual_only"）。
    """
    depth = sample["depth"]

    if depth is None:
        return {"norm_lin": None, "valid_mask": None, "confidence": "none"}

    if not sample["depth_is_metric"]:
        gray = depth.astype(np.float32) / 255.0
        return {
            "norm_lin": gray,
            "valid_mask": gray > 0,
            "confidence": "visual_only",
        }

    norm = normalize_depth(depth)
    valid = depth > 0
    return {
        "norm_lin": norm,
        "valid_mask": valid,
        "confidence": depth_confidence(float(valid.mean())),
    }
