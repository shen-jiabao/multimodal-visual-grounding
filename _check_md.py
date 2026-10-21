# -*- coding: utf-8 -*-
"""校验复现流程.md 的 Markdown 结构（围栏配对、表格空行、各节行数）。"""
import re
from pathlib import Path

P = Path(r"d:\Desktop\2026AiC\答辩材料\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
         r"\AIC-2026-14375345-代码与数据\复现流程.md")
text = P.read_text(encoding="utf-8")
lines = text.splitlines()
fence = text.count("```")
print("总行数", len(lines), "｜代码围栏数", fence, "（偶数即配对）")

bad = [i + 1 for i, ln in enumerate(lines)
       if ln.startswith("|") and i and not (lines[i - 1].startswith("|")
                                           or not lines[i - 1].strip())]
print("表格前缺空行的行号:", bad[:10] or "无")

print("§5 各节：")
for m in re.finditer(r"^### (5\.[0-9a-z]+) (.{0,30})", text, re.M):
    print(f"  {m.group(1):7} {m.group(2)}")
