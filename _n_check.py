#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核 N 件：相对我们最优件（0.7963）到底改了多少、改在哪一层。
用于判断它是不是"单变量（只换模型）+ 限定作用域"的干净件。"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal"
N = SUB / "未提交" / "result_semi_k72b_n519.json"
BEST = SUB / "0.7963" / "result_semi_q38_nonord.json"
CORES = R / "_ordinal_cores_r2.json"


def main() -> int:
    n = json.loads(N.read_text(encoding="utf-8"))
    b = json.loads(BEST.read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))

    print(f"N 件 {len(n)} 条｜最优件 {len(b)} 条｜序数集 {len(cores)}")
    assert set(n) == set(b), "键集不一致"

    ordc = nonc = 0
    moved = []
    for k in n:
        if n[k]["bbox"] != b[k]["bbox"]:
            moved.append(k)
            if k in cores:
                ordc += 1
            else:
                nonc += 1
    print(f"总差异 {len(moved)} 条（预登记 N 记 304）"
          f"｜其中序数 {ordc}｜非序数 {nonc}")
    print(f"  序数层是否逐位不动: {'✓ 是' if ordc == 0 else '★否，动了 ' + str(ordc)}")

    # 字段回显自检
    bad = [k for k in n if any(n[k].get(f) != b[k].get(f)
                              for f in ("visible", "infrared", "depth", "query"))]
    print(f"  四字段回显与最优件一致: {'✓' if not bad else '★异常 ' + str(len(bad))}")
    print(f"  样本差异前 5: {moved[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
