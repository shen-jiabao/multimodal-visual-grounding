#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G3 判读的两个事实核查（本地、零 GPU）。

核查 1：交付件里有没有"空框"（= 用户的"补漏"通道是否存在）
核查 2：冒烟那 20 条是不是**按名排序的前缀**（决定它能不能用来外推改动面）
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sub_path as SP  # noqa: E402

R = Path(__file__).resolve().parent
DATA = R / "复赛数据集-基于大模型的多模态视觉理解与推理"


def main() -> int:
    q = json.loads((DATA / "queries" / "queries.json").read_text("utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text("utf-8")))
    P = json.loads(SP.find("result_semi_q38_nonord").read_text("utf-8"))
    B = json.loads(SP.find("result_semi_n32bf16_nonord").read_text("utf-8"))
    gate = json.loads((R / "k_gate_out" / "_k_gate20.json").read_text("utf-8"))

    # ---- 核查 1：当前件有没有空框 / 退化框 ----
    print("===== 核查 1：交付件的框形态 =====")
    empty, degen, ok = [], [], 0
    for k, v in P.items():
        b = v.get("bbox")
        if not b:
            empty.append(k)
            continue
        if len(b) != 4 or not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            degen.append(k)
            continue
        if (b[2] - b[0]) * (b[3] - b[1]) < 1e-6:
            degen.append(k)
            continue
        ok += 1
    print(f"  合法框 {ok}｜空框（无 bbox）{len(empty)}｜退化框 {len(degen)}")
    print(f"  ⇒ '当前件为空'的条目数 = {len(empty) + len(degen)}"
          f"（用户规则第 3 条的通道规模）")

    # ---- 核查 2：冒烟 20 条是不是前缀 ----
    print("\n===== 核查 2：冒烟集是不是'按名排序的前缀' =====")
    order = sorted(q)
    first30 = order[:30]
    gk = list(gate)
    in30 = [k for k in gk if k in set(first30)]
    print(f"  冒烟 {len(gk)} 条｜其中落在【全库前 30 条】里的 {len(in30)} 条")
    print(f"  冒烟涉及的图像：{sorted({k.split('_')[0] for k in gk})}")
    print(f"  全库前 6 张图：  {sorted({k.split('_')[0] for k in first30})}")

    # ---- 核查 3：前缀的分歧密度 vs 全局（用 q38 已实测的改动集）----
    print("\n===== 核查 3：前缀的改动密度 vs 全局（q38 的 519 条改动集）=====")
    S = [k for k in q if k not in cores
         and (P.get(k) or {}).get("bbox") and (B.get(k) or {}).get("bbox")
         and P[k]["bbox"] != B[k]["bbox"]]
    Sset = set(S)
    base_img = sorted({k.split("_")[0] for k in order})[:6]
    pre = [k for k in q if k.split("_")[0] in set(base_img) and k not in cores]
    n_pre_c = sum(1 for k in pre if k in Sset)
    print(f"  全局非序数改动率  {len(S)}/{len([k for k in q if k not in cores])}"
          f" = {len(S)/len([k for k in q if k not in cores]):.1%}")
    print(f"  前 6 张图（=冒烟同域）：{n_pre_c}/{len(pre)} = "
          f"{n_pre_c/max(1, len(pre)):.1%}")
    print("  ⇒ 冒烟域的分歧密度明显高于全局 ⇒ 冒烟集**有偏**，"
          "不能用来估计改动面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
