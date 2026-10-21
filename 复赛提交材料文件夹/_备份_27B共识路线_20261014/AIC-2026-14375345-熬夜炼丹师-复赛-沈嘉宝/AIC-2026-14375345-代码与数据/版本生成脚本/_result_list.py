#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成提交件（result.zip）相对主件的改动清单，供可视化。"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = R / "复赛提交材料文件夹" / "参赛团队编号-团队名称-复赛-队长姓名"
DO = PKG / "参赛团队编号-代码与数据" / "data_outputs"
ZIP = PKG / "参赛团队编号-赛题结果文档" / "result.zip"

with zipfile.ZipFile(ZIP) as zf:
    sub = json.loads(zf.read("submission.json").decode("utf-8"))
main = json.loads((DO / "predictions_27b.json").read_text("utf-8"))

ks = [k for k in sub if sub[k]["bbox"] != main[k]["bbox"]]
(R / "_result_changed.json").write_text(
    json.dumps(ks, ensure_ascii=False), "utf-8")
print(f"提交件相对主件改动 {len(ks)} 条")
print("前 8 条：", ks[:8])
