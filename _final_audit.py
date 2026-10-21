#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""材料包终检：完整清单、提交件校验、SHA256 一致性、结果唯一性。"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"

print("=" * 70)
print("【1】完整文件清单")
files = sorted(p for p in PKG.rglob("*") if p.is_file())
total = 0
for p in files:
    sz = p.stat().st_size
    total += sz
    print(f"  {sz/1024:>9.1f} KB  {p.relative_to(PKG)}")
print(f"  合计 {len(files)} 个文件，{total/1024/1024:.2f} MB")

print("=" * 70)
print("【2】提交件四项校验（直接读 zip）")
zp = PKG / "AIC-2026-14375345-赛题结果文档" / "result.zip"
with zipfile.ZipFile(zp) as zf:
    names = zf.namelist()
    sub = json.loads(zf.read("submission.json").decode("utf-8"))
q = json.loads((PKG / "AIC-2026-14375345-代码与数据" / "data_outputs"
                / "queries.json").read_text("utf-8"))
print(f"  [1] zip 内部文件 {names}（应仅 1 个）"
      f" → {'OK' if len(names) == 1 else '✗'}")
print(f"  [2] 条数 {len(sub)} / 应 {len(q)} → {'OK' if len(sub) == len(q) else '✗'}")
print(f"  [3] 键集一致 → {'OK' if set(sub) == set(q) else '✗'}")
echo = sum(1 for k, v in sub.items()
           if any(v[f] != q[k][f] for f in ("visible", "infrared", "depth", "query")))
print(f"  [4] 四字段原样回显（不一致 {echo} 条）→ {'OK' if echo == 0 else '✗'}")
bad = [k for k, v in sub.items()
       if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
               and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
print(f"  [5] bbox 合法性（非法 {len(bad)} 条）→ {'OK' if not bad else '✗'}")

print("=" * 70)
print("【3】SHA256 一致性")
sha_txt = (PKG / "AIC-2026-14375345-赛题结果文档" / "SHA256.txt").read_text("utf-8")
want = sha_txt.split()[0].strip()
got = hashlib.sha256(zp.read_bytes()).hexdigest().upper()
print(f"  SHA256.txt 记录：{want}")
print(f"  result.zip 实测：{got}")
print(f"  {'★ 一致' if want == got else '✗ 不一致'}")

print("=" * 70)
print("【4】结果唯一性")
zips = [p.relative_to(PKG) for p in files if p.suffix.lower() == ".zip"]
print(f"  材料包内 zip：{zips}")
preds = [p.relative_to(PKG) for p in files
         if p.name.startswith("predictions")]
print(f"  预测文件：{preds}")
print(f"  提交用结果 zip 数量：{len(zips)}"
      f" → {'OK（唯一）' if len(zips) == 1 else '注意'}")
