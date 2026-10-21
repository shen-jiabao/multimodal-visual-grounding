#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取两份 PDF 的文本，供合规对照（输出到 _pdf_text/）。"""
from __future__ import annotations

from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
OUT = R / "_pdf_text"
OUT.mkdir(exist_ok=True)


def extract(p: Path) -> str:
    """优先 pypdf，回退 PyPDF2 / pdfplumber。"""
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            from pdfplumber import open as popen
            with popen(str(p)) as pdf:
                return "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    r = PdfReader(str(p))
    return "\n".join((pg.extract_text() or "") for pg in r.pages)


for f in ["复赛作品提交要求.pdf", "赛题：基于大模型的多模态视觉理解与推理.pdf",
          r"复赛提交材料文件夹\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝\AIC-2026-14375345-技术方案.PDF"]:
    src = R / f
    if not src.exists():
        print(f"缺文件 {f}")
        continue
    try:
        t = extract(src)
    except Exception as e:
        print(f"{f} 提取失败: {e}")
        continue
    dst = OUT / (src.stem + ".txt")
    with open(dst, "w", encoding="utf-8", errors="replace") as fh:
        fh.write(t)
    print(f"{f} → {dst.name}（{len(t)} 字）")
