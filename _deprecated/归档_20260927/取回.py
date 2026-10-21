# -*- coding: utf-8 -*-
"""一键取回：把本归档目录里的文件按 MANIFEST 搬回**项目根原位**。

为什么必须存在：归档只做"移动 + 记账"，不删除。任何被归档的脚本若要重新运行，
**必须先取回原位**——因为脚本内的路径常量（`Path(r"d:\\Desktop\\2026AiC")` 加文件名）
仍指向项目根，在归档区里跑不起来。

用法
    python -B _deprecated/归档_20260927/取回.py            # 预演（只打印）
    python -B _deprecated/归档_20260927/取回.py --apply    # 真取回
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # _deprecated/归档_x → 项目根
MAN = HERE / "MANIFEST.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真取回（缺省只预演）")
    a = ap.parse_args()
    if not MAN.exists():
        sys.exit(f"缺清单 {MAN}")
    rows = []
    for ln in MAN.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) == 3:
            rows.append((parts[0], parts[1]))
    print(f"清单 {len(rows)} 项 → 目标 {ROOT}")
    n = 0
    for nm, rel in rows:
        is_dir = nm.startswith("[目录]")
        name = nm.replace("[目录] ", "") if is_dir else nm
        src = HERE / rel
        if not src.exists():
            print(f"  跳过（不在本归档）: {name}")
            continue
        tgt = ROOT / name
        if tgt.exists():
            print(f"  ★目标已存在，跳过: {name}")
            continue
        if a.apply:
            shutil.move(str(src), str(tgt))
        n += 1
    print(("已取回 " if a.apply else "[预演] 将取回 ") + f"{n} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())