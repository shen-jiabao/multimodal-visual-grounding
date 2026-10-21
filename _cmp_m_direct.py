# -*- coding: utf-8 -*-
"""比对两份 27B 直出产物：_qwen38_direct.json（交付件直出）与 _m_direct.json（M 件直出）。

回答：它们是同一次跑吗？各自的无框集合是什么？
"""
import hashlib
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = (R / "答辩材料" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
       / "AIC-2026-14375345-代码与数据")
FILES = {
    "_qwen38_direct.json（随包·交付件直出）": PKG / "_qwen38_direct.json",
    "_m_direct.json（本地·M 件直出）": R / "_m_direct.json",
    "_m_direct_filled.json（本地·M 件回填后）": R / "_m_direct_filled.json",
    "_qwen32b_direct_ord.json（随包·32B 直出）": PKG / "_qwen32b_direct_ord.json",
}

data = {}
for name, path in FILES.items():
    raw = path.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    nobox = [k for k, v in d.items()
             if not (isinstance(v, dict) and isinstance(v.get("bbox"), list)
                     and len(v["bbox"]) == 4)]
    data[name] = set(nobox)
    print(f"{name}")
    print(f"    条数 {len(d)}｜无框 {len(nobox)}｜SHA256 {hashlib.sha256(raw).hexdigest()[:16]}…")

names = list(data)
print()
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        print(f"无框集合 {a.split('（')[0]} vs {b.split('（')[0]}: "
              f"相同 {len(data[a] & data[b])}｜仅前者 {len(data[a] - data[b])}｜"
              f"仅后者 {len(data[b] - data[a])}")
