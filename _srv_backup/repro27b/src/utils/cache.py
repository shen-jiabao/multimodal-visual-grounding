"""通用缓存工具：内容哈希 + 有界 LRU（纯 Python / numpy，不依赖 torch）。

用于 Qwen3-VL 的"同图视觉特征缓存"：以图像内容哈希为键，缓存视觉编码结果。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Hashable, Optional

import numpy as np


def content_hash(data: Any) -> str:
    """对图像内容做哈希，作为缓存键。

    支持 ``bytes`` / ``bytearray`` / ``numpy.ndarray`` / 具备 ``tobytes()`` 的对象
    （如 PIL Image）。使用**全量字节**（非采样）以避免相似图碰撞导致误复用。
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
    elif isinstance(data, np.ndarray):
        raw = np.ascontiguousarray(data).tobytes()
    elif hasattr(data, "tobytes"):
        raw = data.tobytes()
    else:
        raw = repr(data).encode("utf-8", "ignore")
    import hashlib
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


class LRUCache:
    """有界 LRU 缓存：``get`` 命中即提升最近使用；``put`` 超限淘汰最旧。"""

    def __init__(self, maxsize: int = 4):
        self.maxsize = max(1, int(maxsize))
        self._data: "OrderedDict[Hashable, Any]" = OrderedDict()

    def get(self, key: Hashable, default: Optional[Any] = None) -> Any:
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return default

    def put(self, key: Hashable, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()
