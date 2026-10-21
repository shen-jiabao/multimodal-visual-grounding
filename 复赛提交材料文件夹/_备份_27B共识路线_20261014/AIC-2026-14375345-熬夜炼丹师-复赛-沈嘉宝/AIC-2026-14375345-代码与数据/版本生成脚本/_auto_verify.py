#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端自动复现验证。

在全新目录里只放入管线必需的 4 个输入，运行后与已提交件比对 SHA256；
并检查投票池是否含"自证"成员（把主件/参照件自己算作投票者）。
"""
from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = (R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝")
CODE = PKG / "AIC-2026-14375345-代码与数据" / "code27b"
DO = PKG / "AIC-2026-14375345-代码与数据" / "data_outputs"
SUBMIT = PKG / "AIC-2026-14375345-赛题结果文档" / "result.zip"

WORK = R / "_auto_test"
if WORK.exists():
    shutil.rmtree(WORK)
(WORK / "code27b" / "scripts").mkdir(parents=True)
(WORK / "data_outputs").mkdir(parents=True)

# —— 只投放管线必需的输入，不带任何其他东西 ——
shutil.copy(CODE / "scripts" / "consensus_vote.py",
            WORK / "code27b" / "scripts" / "consensus_vote.py")
shutil.copy(DO / "consensus_pool.json.gz",
            WORK / "data_outputs" / "consensus_pool.json.gz")
for f in ("predictions_27b.json", "predictions_ref.json"):
    shutil.copy(DO / f, WORK / "data_outputs" / f)

print("干净工作目录内容：")
for p in sorted(WORK.rglob("*")):
    if p.is_file():
        print("  ", p.relative_to(WORK))

r = subprocess.run(
    [sys.executable, str(WORK / "code27b" / "scripts" / "consensus_vote.py")],
    capture_output=True, text=True, encoding="utf-8", errors="replace")
print("\n--- 管线输出 ---")
print(r.stdout.strip())
if r.returncode != 0:
    print("退出码", r.returncode, r.stderr[-500:])

out_zip = WORK / "data_outputs" / "result.zip"
if not out_zip.exists():
    print("未产出 result.zip")
    raise SystemExit(1)

got = hashlib.sha256(out_zip.read_bytes()).hexdigest().upper()
want = hashlib.sha256(SUBMIT.read_bytes()).hexdigest().upper()
print(f"\n复现件 SHA256 {got}")
print(f"提交件 SHA256 {want}")
print("★ 完全一致" if got == want else "✗ 不一致")

# —— 池自证检查 ——
with gzip.open(DO / "consensus_pool.json.gz", "rt", encoding="utf-8") as fh:
    pool = json.load(fh)


def _bb(d):
    return {k: (v["bbox"] if isinstance(v, dict) else v) for k, v in d.items()}


mb = _bb(json.loads((DO / "predictions_27b.json").read_text("utf-8")))
rb = _bb(json.loads((DO / "predictions_ref.json").read_text("utf-8")))
hit_main = [k for k, v in pool.items() if _bb(v) == mb]
hit_ref = [k for k, v in pool.items() if _bb(v) == rb]
print(f"\n池内成员数 {len(pool)}")
print(f"池中与【主件】完全相同的成员：{hit_main if hit_main else '无 ✅'}")
print(f"池中与【参照件】完全相同的成员：{hit_ref if hit_ref else '无 ✅'}")
