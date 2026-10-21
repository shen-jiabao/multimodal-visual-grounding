"""单元测试：通用文本工具 clean_query / clean_prompt_text / split_by_count（不依赖 torch）。

运行：python -m pytest tests/test_text_utils.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from src.utils.text import clean_prompt_text, clean_query, split_by_count

from src.postprocess.bbox import normalize_text


# ---------- clean_query ----------

def test_clean_query_strips_bom_and_zero_width():
    assert clean_query("\ufeffLicense plate A123\ufeff") == "License plate A123"
    assert clean_query("a\u200bb\u200cc\u200dd\u2060e") == "abcde"


def test_clean_query_normalizes_dashes():
    assert clean_query("man\u2011made ground") == "man-made ground"
    assert clean_query("a\u2010b\u2012c\u2013d\u2014e") == "a-b-c-d-e"


def test_clean_query_idempotent_and_plain_untouched():
    src = "The vendor sign \u2011 food\ufeff"
    once = clean_query(src)
    assert clean_query(once) == once
    assert clean_query("A red car.") == "A red car."


# ---------- clean_prompt_text ----------

def test_clean_prompt_text_is_superset_of_clean_query():
    """提示词清洗必须是 clean_query 的严格超集：前者结果 = 后者 + 排版归一。"""
    src = "The vendor sign \u2011 food\ufeff"
    assert clean_prompt_text(src) == clean_query(src)


def test_clean_prompt_text_strips_trailing_whitespace_and_newlines():
    assert clean_prompt_text("A monkey sitting on the ground ") == \
        "A monkey sitting on the ground"
    assert clean_prompt_text("line one\nline two") == "line one line two"
    assert clean_prompt_text("a\t\tb") == "a b"


def test_clean_prompt_text_normalizes_typography():
    assert clean_prompt_text("the woman\u2019s arm") == "the woman's arm"
    assert clean_prompt_text("a\u00a0b") == "a b"
    assert clean_prompt_text("wait\u2026") == "wait..."


def test_clean_prompt_text_preserves_semantics():
    """只做等价改写：不改词序、不改大小写、不动普通 ASCII 文本。"""
    plain = "The second white diamond-shaped sign from bottom to top on the road"
    assert clean_prompt_text(plain) == plain
    assert clean_prompt_text("The leftmost person") == "The leftmost person"


def test_clean_prompt_text_idempotent():
    src = "  the woman\u2019s  arm\ufeff \n"
    once = clean_prompt_text(src)
    assert clean_prompt_text(once) == once


# ---------- normalize_text（匹配层与提示词层同一套清洗） ----------

def test_normalize_text_matches_cleaned_prompt_text():
    """提示词里的 query 已去 BOM/尾部空白，label 回显同形时匹配必须成立。"""
    q = "The license plate of the red car\ufeff"
    assert normalize_text(q) == normalize_text("The license plate of the red car")
    assert normalize_text("The coat on the woman\u2019s arm") == \
        "the coat on the woman's arm"


def test_normalize_text_still_lowercases_and_strips_period():
    assert normalize_text("A Red Car.") == "a red car"
    assert normalize_text("  spaced   out  ") == "spaced out"


# ---------- split_by_count ----------

def test_split_by_count_basic():
    items = [1, 2, 3, 4, 5]
    assert split_by_count(items, 2) == [[1, 2], [3, 4], [5]]
    assert split_by_count(items, 5) == [[1, 2, 3, 4, 5]]
    assert split_by_count(items, 6) == [[1, 2, 3, 4, 5]]


def test_split_by_count_disabled_returns_single_chunk():
    items = [1, 2, 3]
    assert split_by_count(items, None) == [[1, 2, 3]]
    assert split_by_count(items, 0) == [[1, 2, 3]]
    assert split_by_count(items, -3) == [[1, 2, 3]]


def test_split_by_count_empty():
    assert split_by_count([], 2) == []
    assert split_by_count([], None) == []
