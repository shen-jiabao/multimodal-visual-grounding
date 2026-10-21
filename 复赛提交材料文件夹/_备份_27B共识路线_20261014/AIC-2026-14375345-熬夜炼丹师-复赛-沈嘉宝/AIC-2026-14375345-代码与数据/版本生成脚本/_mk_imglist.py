#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 queries.json 抽出被引用的图像文件名，供上传白名单使用。

为什么只传被引用的：本地三个模态目录各 2000 张（合计 12.5 GB），
但 queries.json 只引用其中约 1295 个文件名 ⇒ 传全部会白费一半传输与磁盘。
`_rc.py --sync --only` 按**文件名**过滤，而同一文件名在 visible/infrared/depth
三处都存在 ⇒ 一份白名单正好同时覆盖三个模态。
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
DS = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
MODS = ("visible", "infrared", "depth")


def main() -> int:
    q = json.loads((DS / "queries" / "queries.json").read_text("utf-8"))
    names = sorted({Path(v["visible"]).name for v in q.values()})

    out = R / "_imglist.txt"
    out.write_text("\n".join(names) + "\n", encoding="utf-8")

    print(f"查询条数 {len(q)}｜引用图像名 {len(names)} 个")
    print(f"白名单已写 {out.name}")
    print()
    total = 0
    for m in MODS:
        have = sum(1 for n in names if (DS / "Images" / m / n).exists())
        sz = sum((DS / "Images" / m / n).stat().st_size
                 for n in names if (DS / "Images" / m / n).exists())
        total += sz
        print(f"  {m:9s} 齐备 {have:>4}/{len(names)}   {sz / 2**20:>8.1f} MB")
    print(f"  {'合计':9s} {'':>13}   {total / 2**20:>8.1f} MB")
    print(f"\n对比：传全部三个模态 = 6000 文件 / 12.5 GB"
          f"；只传被引用的 ≈ {len(names) * 3} 文件 / {total / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
