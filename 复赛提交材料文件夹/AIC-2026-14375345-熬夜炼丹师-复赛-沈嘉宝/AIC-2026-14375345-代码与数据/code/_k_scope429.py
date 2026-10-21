#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""切出 K 件各设计的作用域（与 `_assembly_32b.py` 的分桶逐条一致，判据只实现一处）。

    L（B′ 补漏）：作用域 = 桶1（q38 无框）+ 桶3（面积>0.35 被拒）
    N（通道验证）：作用域 = 桶6（q38 改动了的位置，即现状框 = q38 的框）

★ 自检：本脚本复现的分桶必须与谱系记录的 `329 / 0 / 100 / 0 / 3639 / 519`
   **逐桶吻合**，否则说明口径有出入，不得使用其输出。
"""
from __future__ import annotations
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

import json
from collections import Counter
from pathlib import Path

import _fuse_qwen as F      # 几何原语与常量（判据只实现一处）
import _sub_path as SP      # 提交件按名解析（归档目录按得分命名）

R = Path(ROOT_S)
EXPECT = {"1 无框": 329, "2 退化": 0, "3 面积>0.35": 100,
          "4 面积过小": 0, "5 一致": 3639, "6 改动": 519}


def main() -> int:
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text("utf-8")))
    base = json.loads(SP.find("result_semi_n32bf16_nonord").read_text("utf-8"))
    direct = json.loads((R / "_qwen38_direct.json").read_text("utf-8"))
    q = json.loads((R / "复赛数据集-基于大模型的多模态视觉理解与推理"
                    / "queries" / "queries.json").read_text("utf-8"))
    print(f"基线 {SP.find('result_semi_n32bf16_nonord').name}｜"
          f"q38 产物 {len(direct)} 条｜序数 {len(cores)}")

    nonord = [k for k in base if k not in cores]
    st = Counter()
    scope, changes = [], []
    for k in nonord:
        rec = direct.get(k) or {}
        b = rec.get("bbox")
        if not b:
            st["1 无框"] += 1
            scope.append(k)
            continue
        nb = F.norm(b)
        if not F.ordered(nb):
            st["2 退化"] += 1
            continue
        if F.area(nb) > F.MAX_AREA:
            st["3 面积>0.35"] += 1
            scope.append(k)
            continue
        if F.area(nb) < F.MIN_AREA:
            st["4 面积过小"] += 1
            continue
        if F.iou(nb, base[k]["bbox"]) >= F.IOU_KEEP:
            st["5 一致"] += 1
            continue
        st["6 改动"] += 1
        changes.append(k)

    print("\n复现的分桶（与谱系记录逐桶对照）：")
    ok = True
    for key in ["1 无框", "2 退化", "3 面积>0.35", "4 面积过小",
                "5 一致", "6 改动"]:
        got, exp = st[key], EXPECT[key]
        flag = "✓" if got == exp else f"★不符（期望 {exp}）"
        ok &= got == exp
        print(f"  {key:<14}{got:>6}   {flag}")
    tot = sum(st.values())
    print(f"  {'合计':<14}{tot:>6}   {'✓ = 非序数 4587' if tot == len(nonord) else '★'}")
    ok &= tot == len(nonord)

    if not ok:
        print("\n★ 分桶与记录不符 ⇒ 口径有出入，**不得使用**本次输出。")
        return 1

    print(f"\n★ 作用域（与装配器同口径）：")
    out1 = R / "_k_scope429.json"
    out1.write_text(json.dumps(sorted(scope), ensure_ascii=False),
                    encoding="utf-8")
    print(f"  L（补漏，桶1+桶3）        = {len(scope):>5} 条 → {out1.name}")
    out2 = R / "_k_scope519.json"
    out2.write_text(json.dumps(sorted(changes), ensure_ascii=False),
                    encoding="utf-8")
    print(f"  N（通道验证，桶6）        = {len(changes):>5} 条 → {out2.name}")

    P = json.loads(SP.find("result_semi_q38_nonord").read_text("utf-8"))
    same = sum(1 for k in scope if base[k]["bbox"] == P[k]["bbox"])
    print(f"  核查：L 作用域里'当前件 == 基线'的 {same}/{len(scope)}"
          f"（应为全部，因为桶1/桶3 都保留了基线）")
    diff = sum(1 for k in changes if base[k]["bbox"] != P[k]["bbox"])
    print(f"  核查：N 作用域里'当前件 != 基线'的 {diff}/{len(changes)}"
          f"（应为全部，桶6 就是改动集）")
    print(f"  分布检查：L 涉及 {len({k.split('_')[0] for k in scope})} 张图｜"
          f"N 涉及 {len({k.split('_')[0] for k in changes})} 张图")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
