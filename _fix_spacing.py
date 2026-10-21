# -*- coding: utf-8 -*-
"""去除技术文档中符号（/ - +）两侧的多余空格，保持中文排版紧凑。

规则：中文语境下的 "/"、"-"、"+" 两侧不留空格；
数学式（float32 / 255.0）与箭头（→）的间距按惯例保留。
"""
from pathlib import Path

SRC = Path(r"d:\Desktop\2026AiC\技术文档.md")

# (原文, 替换)；每条必须唯一命中，否则报错以便人工核对
RULES: list[tuple[str, str]] = [
    ("语言 - 视觉对齐", "语言-视觉对齐"),
    ("三模态路径 + 查询文本", "三模态路径+查询文本"),
    ("（归一化 / 去噪 / 离群剔除 / 低光融合）", "（归一化/去噪/离群剔除/低光融合）"),
    ("（三级匹配链 + 空间/序数约束排序）", "（三级匹配链+空间/序数约束排序）"),
    ("（多查询 / 图）", "（多查询/图）"),
    ("visible / infrared / depth", "visible/infrared/depth"),
    ("三模态图像路径 + `queries.json`", "三模态图像路径+`queries.json`"),
    ("`AutoModelForImageTextToText` + `AutoProcessor`",
     "`AutoModelForImageTextToText`+`AutoProcessor`"),
    ("（Images/ + queries.json）", "（Images/、queries.json）"),
    ("红外经灰度化 + 中值滤波去噪后", "红外经灰度化+中值滤波去噪后"),
    ("Image 1: visible RGB image / Image 2: infrared thermal image / "
     "Image 3: depth map",
     "Image 1 为可见光、Image 2 为红外热像、Image 3 为深度图"),
    ("同类实例枚举 + 确定性排序", "同类实例枚举+确定性排序"),
    ("取最大 / 最显著目标", "取最大/最显著目标"),
    ("枚举 + 选择", "枚举+选择"),
    ("`high / medium / low / skip`", "`high/medium/low/skip`"),
    ("`closest / nearest / farthest / furthest`",
     "`closest/nearest/farthest/furthest`"),
    ("版本 / 要求", "版本/要求"),
    ("（0 \\~ 19999 毫米）", "（0\\~19999 毫米）"),
]

text = SRC.read_text(encoding="utf-8")
missing: list[str] = []
multi: list[str] = []
for old, new in RULES:
    cnt = text.count(old)
    if cnt == 0:
        missing.append(old)
    elif cnt > 1:
        multi.append(f"{cnt}x {old}")
    else:
        text = text.replace(old, new)

if missing or multi:
    print("未命中:", missing)
    print("多次命中:", multi)
    raise SystemExit(1)

SRC.write_text(text, encoding="utf-8")
print(f"已替换 {len(RULES)} 处")
