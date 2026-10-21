# -*- coding: utf-8 -*-
"""把复赛材料包重新打包为同名 zip（正斜杠路径 + UTF-8 条目名）。

Windows 的 .NET Framework 版 ZipFile.CreateFromDirectory 会用反斜杠作为
条目分隔符，不符合 ZIP 规范，跨平台解压可能异常；改用 Python zipfile 打包。
"""
import os
import zipfile
from pathlib import Path

ROOT = Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹")
NAME = "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
SRC = ROOT / NAME
DST = ROOT / (NAME + ".zip")

# 覆盖旧包：旧包为历史版本，含已删除的脚本
if DST.exists():
    DST.unlink()

count = 0
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    for dir_path, dir_names, file_names in os.walk(SRC):
        dir_names.sort()
        for file_name in sorted(file_names):
            full = Path(dir_path) / file_name
            arc = NAME + "/" + full.relative_to(SRC).as_posix()
            zf.write(full, arc)
            count += 1

print(f"files={count}")
print(f"zip={DST}")
print(f"size={DST.stat().st_size}")
