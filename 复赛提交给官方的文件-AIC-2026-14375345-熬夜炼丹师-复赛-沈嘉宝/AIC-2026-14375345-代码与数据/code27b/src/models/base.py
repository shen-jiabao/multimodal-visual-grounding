"""模型抽象基类：定义统一、可插拔的视觉定位推理接口。

所有视觉定位模型封装都实现本接口，使推理主流程（``src/inference/predictor.py``）
与具体模型实现解耦。

切换模型的唯一入口是 ``configs/config.yaml``：

    model:
      type: "qwen3_vl"            # 换成其它已注册的模型类型
      name_or_path: "..."         # 模型权重路径 / HF 在线 ID

新增模型时：新建一个继承本类的封装文件，并在
``src/models/factory.py`` 的 ``_REGISTRY`` 中登记一行即可。
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.config import Config

logger = logging.getLogger(__name__)


class BaseGroundingModel(ABC):
    """视觉定位模型统一接口。

    子类需实现 :meth:`predict_boxes`，返回统一结构
    （``scores / labels / boxes``），以便复用
    ``inference.matcher`` 的匹配逻辑。
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._resolve_device(cfg.model.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """解析设备字符串：auto -> cuda(若有)/cpu，并做 cuda 可用性兜底。"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            device = "cpu"
        return torch.device(device)

    @abstractmethod
    def predict_boxes(
        self,
        image: np.ndarray,
        texts: List[str],
        box_threshold: float = 0.20,
        text_threshold: float = 0.15,
        extra_images: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """对单张图与多条查询文本做前向，返回检测结果。

        Args:
            image: uint8 RGB 图像 (H, W, 3)。
            texts: 查询文本列表。
            box_threshold: 检测框置信度阈值。
            text_threshold: 文本匹配阈值。
            extra_images: 额外模态图像 ``{名称: uint8 (H,W,3)}``（如 infrared/depth）；
                仅支持多图输入的大模型（Qwen3-VL）使用，检测类模型忽略。

        Returns:
            列表，每个元素形如::

                {
                    "scores": np.ndarray[N] (float32),
                    "labels": list[str]  (长度 N，字符串标签/短语),
                    "boxes":  np.ndarray[N, 4] (float32, 像素坐标 xyxy),
                }

            列表长度 == 前向分块数（通常为 1）。
        """
        raise NotImplementedError