# -*- coding: utf-8 -*-
"""定位 §3.6 校验表所在页与纵向位置（可传入 PDF 路径）。"""
import sys
from pathlib import Path

from pypdf import PdfReader

PKG = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
           r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝")
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    PKG / "AIC-2026-14375345-技术方案.PDF"

KEYS = ["提交件校验与结果", "校验项", "条数", "键集", "字段回显",
        "bbox 合法性", "平台评测指标", "运行配置", "环境与复现",
        "显存", "Ubuntu", "关键依赖", "大文件与外部链接"]
reader = PdfReader(str(PDF))
for pi, page in enumerate(reader.pages, 1):
    found = []
    page.extract_text(visitor_text=lambda t, cm, tm, fd, fs:
                      found.append((round(tm[5], 1), t.strip()))
                      if any(k in t for k in KEYS) else None)
    if found:
        print(f"--- 第 {pi} 页 ---")
        for y, t in found[:14]:
            print(f"   y={y:>7}  {t[:46]}")
print("总页数 =", len(reader.pages))
