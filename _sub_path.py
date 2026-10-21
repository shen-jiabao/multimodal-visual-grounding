# -*- coding: utf-8 -*-
"""提交件路径解析（唯一实现）。

背景：`submission_zips/semifinal` 下的已交件已按**实际得分**归入子目录
（如 `0.7598/result_semi_enum_trunc.zip`），扁平目录只保留 `SHA256.txt`。
原先各脚本硬拼 `SUB / f"{stem}.json"`，归类后即读不到，故统一经此解析。

约定
- `find`      读：先查扁平，再递归各子目录；找不到抛 FileNotFoundError
- `out_path`  写：该件已存在（扁平或某子目录）则**就地覆盖**——避免同一提交件出现
  两份而漂移；否则写扁平目录
"""
from __future__ import annotations

from pathlib import Path

SUB = Path(r"d:\Desktop\2026AiC\submission_zips\semifinal")


def find(stem: str, ext: str = ".json") -> Path:
    """按基名查找提交件（扁平优先，其次递归子目录）。"""
    flat = SUB / f"{stem}{ext}"
    if flat.exists():
        return flat
    hits = sorted(SUB.glob(f"*/{stem}{ext}"))
    if not hits:
        raise FileNotFoundError(
            f"未找到提交件 {stem}{ext}（已查扁平目录与各得分子目录）")
    return hits[0]


def out_path(stem: str, ext: str = ".json") -> Path:
    """写提交件时用：已存在则就地覆盖，否则写扁平目录。"""
    try:
        return find(stem, ext)
    except FileNotFoundError:
        return SUB / f"{stem}{ext}"
