# -*- coding: utf-8 -*-
"""在紧贴代码围栏的表格前补一个空行（渲染更稳妥）。"""
from pathlib import Path

P = Path(r"d:\Desktop\2026AiC\答辩材料\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
         r"\AIC-2026-14375345-代码与数据\复现流程.md")
lines = P.read_text(encoding="utf-8").splitlines()
out, added = [], 0
for i, ln in enumerate(lines):
    if (ln.startswith("|") and out and out[-1].strip() == "```"):
        out.append("")
        added += 1
    out.append(ln)
P.write_text("\n".join(out) + "\n", encoding="utf-8")
print("补空行", added, "处")
