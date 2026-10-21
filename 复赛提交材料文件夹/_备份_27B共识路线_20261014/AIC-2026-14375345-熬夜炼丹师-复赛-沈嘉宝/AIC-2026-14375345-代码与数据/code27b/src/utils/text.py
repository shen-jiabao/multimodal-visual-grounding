"""通用文本处理工具（不依赖 torch）。

用途：
    - :func:`clean_query`：清洗查询/标签中的零宽与特殊字符（BOM、零宽空格、
      异体连字符），供匹配层（``normalize_text``）统一使用。
    - :func:`clean_prompt_text`：提示词侧清洗。在 ``clean_query`` 之上再归一
      排版类字符（弯引号 / 省略号 / 不换行空格 / 全角空格 / 换行制表符），
      用于**送进模型**的 query 文本。此前提示词构造一直用的是**原始**
      ``s.query``，BOM / 零宽字符从未被剥掉。
    - :func:`split_by_count`：按条数切块，供推理层消费
      ``inference.max_queries_per_prompt`` 配置（原为死配置）。
"""
from __future__ import annotations

import re
from typing import List, Sequence, TypeVar

T = TypeVar("T")

# 需删除的零宽 / 不可见字符：BOM、零宽空格、零宽连接/非连接符、word joiner
_ZERO_WIDTH = ("\ufeff", "\u200b", "\u200c", "\u200d", "\u2060")

# 各类连字符（hyphen / non-breaking hyphen / figure dash / en-dash / em-dash）统一为 ASCII '-'
_DASHES = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014")

# 排版类字符 → ASCII 等价物：弯引号、省略号、不换行/窄不换行/全角空格
_TYPOGRAPHY = (
    ("\u2018", "'"), ("\u2019", "'"),
    ("\u201c", '"'), ("\u201d", '"'),
    ("\u2026", "..."),
    ("\u00a0", " "), ("\u202f", " "), ("\u3000", " "),
)

_RE_WS_RUN = re.compile(r"\s+")


def clean_query(text: str) -> str:
    """清洗查询文本中的噪声字符。

    去除 BOM/零宽字符，并把各类连字符统一为 ASCII ``-``。仅做字符级归一，
    不改动词序与语义，可安全地同时用于匹配层的 label 与 query 双方。
    """
    for ch in _ZERO_WIDTH:
        text = text.replace(ch, "")
    for ch in _DASHES:
        text = text.replace(ch, "-")
    return text


def clean_prompt_text(text: str) -> str:
    """提示词侧清洗：在 :func:`clean_query` 之上再归一排版类字符。

    与 ``clean_query`` 的关系是**严格超集**：先跑 ``clean_query``（BOM / 零宽 /
    连字符族），再把弯引号、省略号、不换行与全角空格换成 ASCII 等价物，
    最后把任意空白串（含换行 / 制表符）压成单个空格并去首尾空白。

    只做「等价改写」——不改词序、不改语义、不改大小写，因此不改变任何一条
    query 的含义；它只是把模型需要额外消化的噪声字符提前去掉。

    实测（5690 条复赛 query）：``clean_query`` 会改写 7 条，本函数改写 23 条
    （增量 16 条为尾部空白与弯引号）。
    """
    t = clean_query(text)
    for a, b in _TYPOGRAPHY:
        t = t.replace(a, b)
    return _RE_WS_RUN.sub(" ", t).strip()


def split_by_count(items: Sequence[T], max_count: int | None) -> List[List[T]]:
    """按 ``max_count`` 条数把序列切块。

    ``max_count`` 为 ``None`` 或 ``<= 0`` 时返回单块（即不切分），保证配置
    缺省/非法时行为与"不启用该配置"完全一致；空序列返回空列表。
    """
    seq = list(items)
    if not seq:
        return []
    if max_count is None or max_count <= 0:
        return [seq]
    return [seq[i:i + max_count] for i in range(0, len(seq), max_count)]
