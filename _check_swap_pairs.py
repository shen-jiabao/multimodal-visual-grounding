#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定 swap 系列是否为「成对置换」（算法行为）而非零散指定修改。

判据：若 A 相对 B 的差异位中，A 的框普遍能在 B 的差异位里找到来源
（即两个位置互换），则为置换操作；若差异位零散、框也不成对，则更像人工指选。
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"


def load(name: str) -> dict:
    for p in SEMI.glob(f"**/{name}.json"):
        if "复赛提交材料" in str(p):
            continue
        obj = json.loads(p.read_text("utf-8"))
        out = {}
        for k, v in obj.items():
            b = v.get("bbox") if isinstance(v, dict) else v
            if isinstance(b, (list, tuple)) and len(b) == 4:
                out[str(k)] = [round(float(x), 4) for x in b]
        return out
    raise FileNotFoundError(name)


base = load("result_final_semi")
for name in ("result_v2_swap33", "result_v3_swap58"):
    d = load(name)
    common = [k for k in d if k in base]
    diff = [k for k in common if d[k] != base[k]]
    # base 的差异位框 → 位置集合
    src = {}
    for k in diff:
        src.setdefault(tuple(base[k]), []).append(k)
    paired = [k for k in diff if tuple(d[k]) in src]
    print(f"\n{name}")
    print(f"  与 final_semi 差异 {len(diff)} 条")
    print(f"  其中 {len(paired)} 条的框来自对方的差异位（成对置换）")
    print(f"  置换覆盖率：{len(paired)}/{len(diff)}"
          f" = {100*len(paired)/max(1,len(diff)):.0f}%")
    print("  前 6 个差异位示例：")
    for k in diff[:6]:
        tag = "成对" if k in paired else "零散"
        print(f"    {k}  [{tag}]  base={base[k]}  ->  {d[k]}")
