# -*- coding: utf-8 -*-
"""脚本目录自检：数量统计 + 是否残留悬空的本地导入。"""
import ast
import sys
from pathlib import Path

GEN = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹"
           r"\AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
           r"\AIC-2026-14375345-代码与数据\版本生成脚本")
files = sorted(GEN.glob("*.py"))
stems = {p.stem for p in files}
print(f"脚本总数 = {len(files)}"
      f"（_*.py {sum(1 for p in files if p.name.startswith('_'))} 个，"
      f"*.py {sum(1 for p in files if not p.name.startswith('_'))} 个）")

THIRD = {"numpy", "torch", "PIL", "cv2", "yaml", "tqdm", "transformers",
         "requests", "modelscope", "pypdf", "pdfplumber", "fpdf", "openai",
         "groundingdino", "torchvision", "datasets", "accelerate",
         "safetensors", "scipy", "pandas"}
missing = {}
syntax = []
for path in files:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as ex:
        syntax.append((path.name, str(ex)))
        continue
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods = [node.module.split(".")[0]]
        for m in mods:
            if (m not in sys.stdlib_module_names and m not in THIRD
                    and m not in stems):
                missing.setdefault(m, set()).add(path.name)

print("语法错误 =", syntax or "无 ✔")
print("悬空的本地导入 =", {k: sorted(v) for k, v in missing.items()} or "无 ✔")
