# -*- coding: utf-8 -*-
"""比对复赛主件/参照件与 PRO 6000 实例产出的 M 直出产物是否同源。"""
import hashlib
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = (R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
       / "AIC-2026-14375345-代码与数据" / "data_outputs")
FILES = {
    "复赛主件 predictions_27b.json": PKG / "predictions_27b.json",
    "复赛参照件 predictions_ref.json": PKG / "predictions_ref.json",
    "M 直出(回填后) _m_direct_filled.json": R / "_m_direct_filled.json",
    "M 直出(原始) _m_direct.json": R / "_m_direct.json",
}


def box(v):
    if isinstance(v, dict):
        v = v.get("bbox")
    if isinstance(v, (list, tuple)) and len(v) == 4:
        return [round(float(x), 4) for x in v]
    return None


data = {}
for name, path in FILES.items():
    raw = path.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    data[name] = {k: box(v) for k, v in d.items()}
    nb = sum(1 for v in data[name].values() if v is None)
    print(f"{name}: 条数 {len(d)}｜无框 {nb}｜SHA256 {hashlib.sha256(raw).hexdigest()[:16]}…")

names = list(data)
print()
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = data[names[i]], data[names[j]]
        if set(a) != set(b):
            print(f"{names[i][:12]} vs {names[j][:12]}: 键集不同")
            continue
        same = sum(1 for k in a if a[k] == b[k])
        print(f"{names[i][:14]} vs {names[j][:14]}: 逐位相同 {same}/{len(a)}")
