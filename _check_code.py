# -*- coding: utf-8 -*-
"""核验嵌入的代码：是否完整（含最长提示词行）、是否超出页宽。"""
import importlib.util
import re
from pathlib import Path

from fpdf import FPDF
from pypdf import PdfReader

PKG = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
           r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝")
PDF = PKG / "AIC-2026-14375345-技术方案.PDF"
GEN = PKG / "AIC-2026-14375345-代码与数据/版本生成脚本"
SRC = PKG / "AIC-2026-14375345-代码与数据/code27b"

reader = PdfReader(str(PDF))
text = "".join(p.extract_text() or "" for p in reader.pages)
flat = re.sub(r"\s+", "", text)

print("页数 =", len(reader.pages))
for k in ["def match_by_position", "def build_query_prompt",
          "def parse_spatial_constraint",
          "def select_index", "def _iou", "sa == 0 and sq == m",
          "zipfile.ZipInfo", "def chain_of", "def enforce_ordinal_chains",
          "pava_non_decreasing", "groups.setdefault",
          "assert 0 <= b[0] < b[2] <= 1"]:
    print(("OK  " if re.sub(r"\s+", "", k) in flat else "MISS"), k)

# 最长提示词行是否完整落入 PDF（折行后拼接应能逐字对上）
lines = (SRC / "src/models/qwen3_vl_parse.py").read_text(
    encoding="utf-8").splitlines()
longest = max(lines[47:66], key=len)
probe = re.sub(r"\s+", "", longest)[:180]
print("最长行字符数 =", len(longest), "| 前 180 字完整入文:",
      probe in flat)

# 折行保证：任何折行片段都不超过可用宽度
spec = importlib.util.spec_from_file_location("md2pdf", GEN / "_md2pdf.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
pdf = FPDF()
pdf.add_font("hei", "", "C:/Windows/Fonts/simhei.ttf")
pdf.set_margins(17, 16, 17)
pdf.add_page()
pdf.set_font("hei", size=8.2)
limit = pdf.epw - 6
worst = 0.0
for rel in ["scripts/consensus_vote.py", "src/inference/matcher.py",
            "src/inference/detect_select.py",
            "src/models/qwen3_vl_parse.py"]:
    for ln in (SRC / rel).read_text(encoding="utf-8").splitlines():
        for part in m.wrap_code(pdf, ln, limit):
            worst = max(worst, pdf.get_string_width(part))
print(f"折行后最宽片段 = {worst:.1f}pt / 允许 {limit:.1f}pt ->",
      "未溢出 ✔" if worst <= limit + 0.01 else "溢出 ✘")
