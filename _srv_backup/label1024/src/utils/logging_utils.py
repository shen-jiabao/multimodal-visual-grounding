"""日志工具。

提供统一的日志配置函数，避免各脚本重复配置 logging。
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", name: str = "multimodal_grounding") -> logging.Logger:
    """在 root logger 上配置 stdout handler，返回指定名称的 logger。

    各模块通过 ``logging.getLogger(__name__)`` 创建的 logger 默认向 root
    传播，因此统一配置 root 即可让全部模块的日志可见（避免只配置单个
    具名 logger 导致其它模块 INFO 日志被丢弃）。
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)

    logger = logging.getLogger(name)
    logger.setLevel(root.level)
    logger.propagate = True
    return logger
