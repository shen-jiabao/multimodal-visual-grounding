# -*- coding: utf-8 -*-
"""核验流程图框图：绘制算子 + 所在页与相邻段落位置。"""
import re
import zipfile
from pathlib import Path

from pypdf import PdfReader

PKG = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
           r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝")
PDF = PKG / "AIC-2026-14375345-技术方案.PDF"
DOCX = PKG / "技术文档：基于大模型的多模态视觉理解与推理.docx"

reader = PdfReader(str(PDF))
print("pages", len(reader.pages))
for idx, page in enumerate(reader.pages, 1):
    raw = page.get_contents().get_data()
    text = (page.extract_text() or "").replace("\n", " ")
    curves = raw.count(b" c\n")
    fills = raw.count(b" f\n")
    step1 = "[1] 输入" in text
    tail = "第 1–5 步" in text
    print(f"  p{idx}: 圆角曲线={curves} 箭头填充={fills} "
          f"框1={step1} 后续正文={tail}")
    if step1 or tail:
        print("      末尾:", text[-72:])

xml = zipfile.ZipFile(str(DOCX)).read("word/document.xml").decode(
    "utf-8", "ignore")
print("docx 框图单格表数 =",
      xml.count('w:val="TableGrid"') - 5, "| 箭头 ▼ 数 =", xml.count("▼"))
