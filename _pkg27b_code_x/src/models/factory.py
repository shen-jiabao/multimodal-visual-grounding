"""模型工厂：根据配置创建对应的视觉定位模型。

新增模型时：
    1. 新建一个继承 ``BaseGroundingModel`` 的封装文件；
    2. 在此处 import 并登记到 ``_REGISTRY``；
    3. 在 ``configs/config.yaml`` 里把 ``model.type`` 改成对应键名。
"""
from __future__ import annotations

from src.config import Config
from src.models.base import BaseGroundingModel
from src.models.qwen3_vl import Qwen3VLModel

# 模型类型名 -> 实现类
_REGISTRY = {
    "qwen3_vl": Qwen3VLModel,
}


def create_model(cfg: Config) -> BaseGroundingModel:
    """按 ``cfg.model.type`` 实例化模型。"""
    model_type = cfg.model.type
    if model_type not in _REGISTRY:
        raise ValueError(
            f"未知模型类型 {model_type!r}，可用类型：{sorted(_REGISTRY)}。"
            "请在 configs/config.yaml 的 model.type 中指定已注册的类型。"
        )
    return _REGISTRY[model_type](cfg)