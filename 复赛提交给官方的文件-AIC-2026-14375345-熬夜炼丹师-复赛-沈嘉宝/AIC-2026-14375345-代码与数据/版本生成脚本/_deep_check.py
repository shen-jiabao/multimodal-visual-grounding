#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深度复查：压缩包内部、PDF 元数据、数据文件结构。"""
from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

PKG = (Path(r"d:\Desktop\2026AiC\复赛提交材料文件夹")
       / "参赛团队编号-团队名称-复赛-队长姓名")
CODE = PKG / "参赛团队编号-代码与数据" / "code27b"
DO = PKG / "参赛团队编号-代码与数据" / "data_outputs"

print("=" * 60)
print("【1】投票池内部")
with gzip.open(CODE / "consensus_pool.json.gz", "rt", encoding="utf-8") as fh:
    pool = json.load(fh)
print(f"  成员数：{len(pool)}")
print(f"  key 列表：{list(pool)}")
bad = [k for k in pool if not k.startswith("voter_")]
print(f"  非 voter_ 前缀的 key：{bad if bad else '无 ✅'}")
lens = sorted({len(v) for v in pool.values()})
print(f"  各成员条目数：{lens}")

print("=" * 60)
print("【2】zip 包内容")
for z, label in [(PKG / "参赛团队编号-赛题结果文档" / "result.zip", "赛题结果文档/result.zip"),
                 (DO / "result.zip", "data_outputs/result.zip"),
                 (DO / "result_semi_27b_基线存档.zip", "基线存档.zip")]:
    if not z.exists():
        print(f"  {label}：不存在")
        continue
    with zipfile.ZipFile(z) as zf:
        infos = zf.infolist()
        detail = [(i.filename, i.file_size, i.date_time) for i in infos]
    print(f"  {label} → {detail}")

print("=" * 60)
print("【3】技术方案 PDF 元数据")
try:
    from pypdf import PdfReader
    r = PdfReader(str(PKG / "参赛团队编号-技术方案.PDF"))
    print(f"  页数：{len(r.pages)}")
    print(f"  元数据：{dict(r.metadata) if r.metadata else '（无）'}")
except Exception as e:
    print(f"  读取失败：{e}")

print("=" * 60)
print("【4】数据文件结构抽查")
for name in ["predictions_27b.json", "predictions_ref.json",
             "predictions_consensus.json", "predictions_27b_1004.json",
             "queries.json"]:
    p = DO / name
    if not p.exists():
        print(f"  {name}：不存在")
        continue
    obj = json.loads(p.read_text("utf-8"))
    v = next(iter(obj.values()))
    keys = list(v) if isinstance(v, dict) else "纯 bbox"
    print(f"  {name}: {len(obj)} 条｜value 类型 {type(v).__name__}｜字段 {keys}")
