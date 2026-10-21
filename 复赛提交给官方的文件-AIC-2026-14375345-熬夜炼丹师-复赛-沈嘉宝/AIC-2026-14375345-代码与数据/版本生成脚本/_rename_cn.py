#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 ASCII 打包的中文目录名恢复为中文，并归位到 label1024/scripts/。"""
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC\_srv_backup")
pack = R / "cn_pack"
dst = R / "label1024" / "scripts"
mapping = {"CN_1": "复赛", "CN_2": "初赛"}

for ascii_name, cn_name in mapping.items():
    src = pack / ascii_name
    if not src.exists():
        print(f"缺少 {ascii_name}")
        continue
    target = dst / cn_name
    if target.exists():
        print(f"{cn_name} 已存在，跳过")
        continue
    src.rename(target)
    print(f"{ascii_name} -> {cn_name}")

print("label1024/scripts 下现有目录：")
for p in sorted(dst.iterdir()):
    print("  ", p.name, "（目录）" if p.is_dir() else "")
