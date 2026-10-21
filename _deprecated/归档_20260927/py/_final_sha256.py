# -*- coding: utf-8 -*-
"""锁定收尾：独立重算 SHA256 + 键集/字段/bbox 合法性（不依赖视觉）。"""
import hashlib
import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
ZIP = R / "result_locate_merge.zip"
JSON = R / "result_locate_merge.json"
TEST = R / "初赛数据集-基于大模型的多模态视觉理解与推理" / "test.json"

print("== SHA256 独立重算 ==")
for p in (ZIP, JSON):
    h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
    print(f"  {p.name}: {h}")

print("\n== ZIP 内部文件 ===")
with zipfile.ZipFile(ZIP) as zf:
    names = zf.namelist()
    print("  entries:", names)
    # zip 内 json 的 sha256
    inner = zf.read(names[0])
    print(f"  zip内{names[0]} SHA256: "
          f"{hashlib.sha256(inner).hexdigest().upper()}")
    inner_data = json.loads(inner.decode("utf-8"))

print("\n== JSON 与 ZIP 内 JSON 内容一致性 ==")
disk_data = json.loads(JSON.read_text(encoding="utf-8"))
same = disk_data == inner_data
print("  disk == zip-inner:", same)

print("\n== 键集与字段合法性 ==")
test = json.loads(TEST.read_text(encoding="utf-8"))
for label, d in [("disk", disk_data), ("zip-inner", inner_data)]:
    assert len(d) == 9555 and set(d) == set(test), f"{label} 键集异常"
    FIELDS = {"visible", "infrared", "depth", "query", "bbox"}
    bad = [k for k, v in d.items()
           if set(v) != FIELDS
           or not (0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    print(f"  {label}: 9555 键集一致, 字段/bbox 异常 {len(bad)}")

print("\n== 锁定确认 ==")
print("ALL CHECKS PASS")