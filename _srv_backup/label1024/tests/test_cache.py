"""单元测试：通用缓存工具 LRUCache / content_hash（不依赖 torch）。

运行：python -m pytest tests/test_cache.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

import numpy as np

from src.utils.cache import LRUCache, content_hash


# ---------------- content_hash ----------------

def test_content_hash_deterministic():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    assert content_hash(a) == content_hash(a.copy())


def test_content_hash_differs_for_different_content():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.ones((4, 4, 3), dtype=np.uint8)
    assert content_hash(a) != content_hash(b)


def test_content_hash_accepts_bytes_and_tobytes_obj():
    class _HasTobytes:
        def tobytes(self):
            return b"abc"

    assert content_hash(b"abc") == content_hash(_HasTobytes())
    assert content_hash(bytearray(b"abc")) == content_hash(b"abc")


# ---------------- LRUCache ----------------

def test_lru_basic_get_put():
    c = LRUCache(maxsize=2)
    c.put("a", 1)
    assert c.get("a") == 1
    assert c.get("missing") is None
    assert c.get("missing", -1) == -1
    assert len(c) == 1
    assert "a" in c


def test_lru_evicts_oldest():
    c = LRUCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)          # 淘汰 a
    assert "a" not in c
    assert c.get("b") == 2 and c.get("c") == 3
    assert len(c) == 2


def test_lru_get_promotes_recent():
    c = LRUCache(maxsize=2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1   # a 变为最近使用
    c.put("c", 3)            # 应淘汰 b（而非 a）
    assert "a" in c and "b" not in c


def test_lru_update_existing():
    c = LRUCache(maxsize=2)
    c.put("a", 1)
    c.put("a", 99)
    assert c.get("a") == 99
    assert len(c) == 1


def test_lru_min_maxsize_and_clear():
    c = LRUCache(maxsize=0)   # 归一到 1
    assert c.maxsize == 1
    c.put("a", 1)
    c.put("b", 2)
    assert len(c) == 1 and "b" in c
    c.clear()
    assert len(c) == 0
