# -*- coding: utf-8 -*-
"""核验表格：同一行内各单元格是否为等高矩形（坐标含负号）。"""
import re
from collections import defaultdict
from pathlib import Path

from pypdf import PdfReader

PKG = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
           r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝")
PDF = PKG / "AIC-2026-14375345-技术方案.PDF"

raw = b"".join(p.get_contents().get_data()
               for p in PdfReader(str(PDF)).pages)
NUM = rb"(-?\d+(?:\.\d+)?)"
rows = defaultdict(list)
for m in re.finditer(NUM + b" " + NUM + b" " + NUM + b" " + NUM + b" re B",
                     raw):
    x, y, w, h = (round(float(g), 2) for g in m.groups())
    rows[y].append((x, abs(w), abs(h)))

print("单元格矩形总数 =", sum(len(v) for v in rows.values()),
      "| 表格行数 =", len(rows))
bad = {y: sorted({t[2] for t in v}) for y, v in rows.items()
       if len({t[2] for t in v}) > 1}
print("行内高度不一致的行:", bad if bad else "无 ✔")

y0 = sorted(rows)[-1]
print("最后一行 y =", y0, " 各单元格 (x, 宽, 高):")
for x, w, h in sorted(rows[y0]):
    print(f"   x={x:>6} 宽={w:>6} 高={h}")
