# -*- coding: utf-8 -*-
"""终验：逐个文本块检查其取色算子，确认全部为纯黑（0 g / 0 0 0 rg）。"""
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

PKG = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
           r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝")
PDF = PKG / "AIC-2026-14375345-技术方案.PDF"

gray = Counter()   # 文本块内的灰度取色   X g
rgb = Counter()    # 文本块内的 RGB 取色  X Y Z rg
for pg in PdfReader(str(PDF)).pages:
    raw = pg.get_contents().get_data()
    for blk in re.findall(rb"q BT (.*?) Tj", raw, re.S):
        for v in re.findall(rb"([\d.]+) g ", blk):
            gray[v.decode()] += 1
        for v in re.findall(rb"([\d.]+ [\d.]+ [\d.]+) rg ", blk):
            rgb[v.decode()] += 1

print("文本块灰度取色分布:", dict(gray))
print("文本块 RGB 取色分布:", dict(rgb) or "无（未使用 RGB 取色）")
ok = set(gray) <= {"0"} and not rgb
print("结论:", "全部文字为纯黑 ✔" if ok else "存在非黑文字 ✘")
