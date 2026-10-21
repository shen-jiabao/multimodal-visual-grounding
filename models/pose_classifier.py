# -*- coding: utf-8 -*-
"""检测后姿态分类器（正/侧/背判别）。

方案 2（两阶段判别）：Grounding DINO 先定位候选框，本模块再对每个候选框
裁剪 RGB/红外/深度三模态 patch，送入轻量 CNN 输出朝向类别
front / side / back，用于区分「正面/侧面」这类细粒度属性。

输入 patch：RGB(3) + 红外(1) + 深度(1) = 5 通道，统一 resize 到固定尺寸。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

# 朝向类别索引
POSE_CLASSES = ["front", "side", "back"]

# 深度归一化范围（与主推理一致，单位 mm）
DEPTH_MIN_MM = 300
DEPTH_MAX_MM = 19999

# RGB 归一化（ImageNet）
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)
# 红外单通道归一化（与 predict_gd_3modal._to_single_tensor 一致）
IR_MEAN = 0.485
IR_STD = 0.229


def _imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR):
    """兼容中文路径读取。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, flags)


def load_ir_gray(path: str) -> np.ndarray:
    """读取红外图并转为单通道灰度（原分辨率）。"""
    img = _imread_unicode(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = _imread_unicode(path, cv2.IMREAD_COLOR)
    assert img is not None, f"红外读取失败: {path}"
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def parse_pose_word(caption: str) -> Optional[int]:
    """从 query 解析朝向词，返回 POSE_CLASSES 索引；无朝向词返回 None。

    0=front, 1=side, 2=back。只匹配明确的「朝向/视角」语义，避免把
    "side of the road"（方位）、"back of the sign"（部位）这类空间描述误判为姿态。
    """
    c = caption.lower().replace("-", " ").replace("_", " ")
    # 背对（back）：面向/背部朝向相机
    if any(k in c for k in [
        "back to the camera", "facing away", "turned away",
        "seen from behind", "rear view", "backwards",
    ]):
        return 2
    # 侧面（side）：明确的侧向/视角词。数据集里侧向多用方向性 facing 表达——
    # 物体正面朝左/右时，相机看到的是其侧面轮廓，故纳入 facing left/right 与
    # turned left/right 两类。（"side view" 会误中 side-view mirror、
    # "side facing" 会误中 "left side facing right"，均已剔除）
    if any(k in c for k in [
        "sideways", "side profile", "profile view", "in profile",
        "seen from the side", "lateral view",
        "facing left", "facing right",
        "facing to the left", "facing to the right",
        "turned to the left", "turned to the right",
        "turned left", "turned right",
    ]):
        return 1
    # 正面（front）：面向/朝向相机或正前方（"head on" 会误中 "head one"，已剔除）
    if any(k in c for k in [
        "facing the camera", "face the camera", "faces the camera",
        "facing the viewer", "faces the viewer", "facing forward",
        "facing directly", "directly facing", "front facing", "front-facing",
        "front view", "frontal",
    ]):
        return 0
    return None


def _crop_box(arr: np.ndarray, box: np.ndarray, h: int, w: int) -> np.ndarray:
    """按归一化 xyxy 裁剪原分辨率图像，非法框退化为整图。"""
    x1, y1, x2, y2 = box
    x1i = int(round(x1 * w))
    x2i = int(round(x2 * w))
    y1i = int(round(y1 * h))
    y2i = int(round(y2 * h))
    x1i = max(0, x1i)
    y1i = max(0, y1i)
    x2i = min(w, x2i)
    y2i = min(h, y2i)
    if x2i <= x1i + 1:
        x1i, x2i = 0, w
    if y2i <= y1i + 1:
        y1i, y2i = 0, h
    return arr[y1i:y2i, x1i:x2i]


def prepare_patch(
    rgb: np.ndarray,
    ir_gray: np.ndarray,
    depth: np.ndarray,
    box: np.ndarray,
    size: int = 224,
) -> torch.Tensor:
    """裁剪并对齐三模态，输出 (5, size, size) float32 patch。

    rgb: (H, W, 3) RGB uint8；ir_gray: (H, W) uint8；depth: (H, W) uint16；
    box: (4,) 归一化 xyxy。
    """
    h, w = rgb.shape[:2]

    # RGB 裁剪 -> resize -> ImageNet 归一化
    rgb_c = _crop_box(rgb, box, h, w)
    rgb_c = cv2.resize(rgb_c, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb_t = rgb_c.astype(np.float32) / 255.0
    mean = np.array(RGB_MEAN, dtype=np.float32)
    std = np.array(RGB_STD, dtype=np.float32)
    rgb_t = (rgb_t - mean) / std
    rgb_t = np.transpose(rgb_t, (2, 0, 1))  # (3, size, size)

    # 红外单通道
    ir_c = _crop_box(ir_gray, box, h, w)
    ir_c = cv2.resize(ir_c, (size, size), interpolation=cv2.INTER_LINEAR)
    ir_t = np.asarray(ir_c, dtype=np.float32) / 255.0
    ir_t = (ir_t - IR_MEAN) / IR_STD
    ir_t = ir_t[np.newaxis, ...]  # (1, size, size)

    # 深度单通道
    d_c = _crop_box(depth, box, h, w).astype(np.float32)
    d_c = cv2.resize(d_c, (size, size), interpolation=cv2.INTER_NEAREST)
    d_c = np.clip(d_c, DEPTH_MIN_MM, DEPTH_MAX_MM)
    d_c = (d_c - DEPTH_MIN_MM) / float(DEPTH_MAX_MM - DEPTH_MIN_MM)
    d_t = d_c[np.newaxis, ...]

    patch = np.concatenate([rgb_t, ir_t, d_t], axis=0)  # (5, size, size)
    return torch.from_numpy(patch.astype(np.float32))


class PoseClassifier(nn.Module):
    """轻量三模态姿态分类器：5 通道输入 -> 3 类朝向 logits。"""

    def __init__(self, in_chans: int = 5, num_classes: int = 3, base: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_chans, base, 7, 2, 3), nn.BatchNorm2d(base),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base, base * 2, 3, 1, 1), nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(base * 2, base * 4, 3, 1, 1), nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, 1, 1), nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 8, 3, 1, 1), nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def predict_pose(
    model: nn.Module,
    patch: torch.Tensor,
    device: torch.device | None = None,
) -> Tuple[int, float]:
    """单 patch 推理，返回 (类别索引, 置信度)。"""
    model.eval()
    d = device or next(model.parameters()).device
    with torch.no_grad():
        logits = model(patch.unsqueeze(0).to(d))
        prob = logits.softmax(dim=1)
        cls = int(prob.argmax(dim=1).item())
        conf = float(prob[0, cls].item())
    return cls, conf