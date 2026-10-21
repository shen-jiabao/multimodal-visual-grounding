# -*- coding: utf-8 -*-
"""27B 共识路线的主件/参照件，与 PRO 6000 实例的 M 直出、I 件直出比对是否同源。"""
import hashlib
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
DO = (R / "复赛提交材料文件夹" / "_备份_27B共识路线_20261014"
      / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
      / "AIC-2026-14375345-代码与数据" / "data_outputs")
FILES = {
    "路线27B 主件 predictions_27b": DO / "predictions_27b.json",
    "路线27B 参照件 predictions_ref": DO / "predictions_ref.json",
    "PRO6000 M 回填后 _m_direct_filled": R / "_m_direct_filled.json",
    "PRO6000 M 原始 _m_direct": R / "_m_direct.json",
    "I 件直出 _qwen38_direct": R / "复赛提交材料文件夹"
    / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
    / "AIC-2026-14375345-代码与数据" / "data_outputs" / "_qwen38_direct.json",
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
        keys = set(a) & set(b)
        if len(keys) < 100:
            print(f"{names[i][:16]} vs {names[j][:16]}: 共享键 {len(keys)} ⇒ 不可比")
            continue
        same = sum(1 for k in keys if a[k] == b[k])
        print(f"{names[i][:16]} vs {names[j][:16]}: 共享键 {len(keys)}｜逐位相同 {same}")
