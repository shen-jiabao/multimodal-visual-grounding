#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线预演：519 条改动里有多少是"模型共识"，多少是"本件孤例"。

为什么要算
    用户问"融合能不能提分"。融合的常见形态是**共识过滤**（只接受多个模型都改的
    条目）。它是否可行，取决于改动面的**成分**——但注意：

    ★ 本脚本只能给"成分"，**给不了"成分各自的正确率"**——因为 queries.json
      没有 answer/gold，本项目原理上不可观测。要分解正确率必须用**归零法**去"测"
      （见文末提示）。不要把本脚本的输出读成"共识那部分更准"。

三个数据源
    P   本件（0.7963）：其非序数框 = q38 的框（在被改的 519 条上）
    B   基线（0.7786）：其非序数框 = 32B-bf16 的框
    A   32B-AWQ 直出（_qwen32b_direct_ord.json，全 5690）——第三个来源

★ 独立性声明：A 与 B **同一模型不同精度**（AWQ int4 vs bf16），
  不是真正的独立第三方；只能说"32B 家族"。这一点会削弱结论强度，必须写明。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sub_path as SP  # noqa: E402

R = Path(__file__).resolve().parent

AGREE = 0.5   # 与装配器的 IOU_KEEP 同口径


def iou(a: list[float], b: list[float]) -> float:
    """两个归一化框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    s = ar + br - inter
    return inter / s if s > 0 else 0.0


def qs(vs: list[float]) -> str:
    """打印分位。"""
    if not vs:
        return "—"
    v = sorted(vs)
    n = len(v)
    return (f"p10={v[n // 10]:.3f} p50={v[n // 2]:.3f} "
            f"p90={v[9 * n // 10]:.3f}  ≥0.5 占 {sum(x >= AGREE for x in v) / n:.1%}")


def analyse(tag: str, ref: dict, q: dict, cores: set, P: dict, B: dict) -> None:
    """以 ref 为参照，看改动面 519 的成分。

    注意：若 ref 与 B 同族（如 32B-AWQ vs 32B-bf16），A≈B 恒成立，
    分类会退化为同义反复 —— 故必须用**真正独立**的 ref。
    """
    nonord = [k for k in q if k not in cores]
    S = [k for k in nonord
         if (P.get(k) or {}).get("bbox") and (B.get(k) or {}).get("bbox")
         and P[k]["bbox"] != B[k]["bbox"]]
    K = [k for k in nonord if k not in set(S)]
    print(f"\n{'=' * 62}\n参照 = {tag}｜非序数 {len(nonord)}｜改动 {len(S)}｜未改动 {len(K)}")

    cal = [iou(B[k]["bbox"], ref[k]["bbox"]) for k in K
           if (ref.get(k) or {}).get("bbox")]
    print(f"  标定（未改动区，{tag} vs 基线）：\n    {qs(cal)}")

    st = {"共识（本件≈ref 且 基线≉ref）": 0, "本件孤例（ref≈基线）": 0,
          "三方分歧（都 <0.5）": 0, "退化（三方都≈）": 0}
    for k in S:
        a = (ref.get(k) or {}).get("bbox")
        if not a:
            st["三方分歧（都 <0.5）"] += 1
            continue
        iq = iou(P[k]["bbox"], a) >= AGREE
        ib = iou(B[k]["bbox"], a) >= AGREE
        if iq and not ib:
            st["共识（本件≈ref 且 基线≉ref）"] += 1
        elif ib and not iq:
            st["本件孤例（ref≈基线）"] += 1
        elif not iq and not ib:
            st["三方分歧（都 <0.5）"] += 1
        else:
            st["退化（三方都≈）"] += 1
    print(f"  改动面成分（IoU≥{AGREE}）：")
    for name, n in st.items():
        print(f"    {name:28s} {n:>4}  {n / max(1, len(S)):>6.1%}")
    assert sum(st.values()) == len(S), "分类不互斥"

    dist = [iou(P[k]["bbox"], ref[k]["bbox"]) for k in S
            if (ref.get(k) or {}).get("bbox")]
    print(f"  改动区 IoU(本件, {tag})：\n    {qs(dist)}")

    # ★ 可判定量：ref 是否"站在本件这边"——ref 与本件的距离 < ref 与基线的距离
    side = 0
    for k in S:
        a = (ref.get(k) or {}).get("bbox")
        if not a:
            continue
        if iou(P[k]["bbox"], a) > iou(B[k]["bbox"], a):
            side += 1
    print(f"  ★ ref 更靠近【本件】而非基线：{side}/{len(S)} = {side / len(S):.1%}")


def main() -> int:
    q = json.loads((R / "复赛数据集-基于大模型的多模态视觉理解与推理"
                    / "queries" / "queries.json").read_text("utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text("utf-8")))
    P = json.loads(SP.find("result_semi_q38_nonord").read_text("utf-8"))
    B = json.loads(SP.find("result_semi_n32bf16_nonord").read_text("utf-8"))

    refs = [
        ("32B-AWQ（★与基线同族，仅供对照）",
         json.loads((R / "_qwen32b_direct_ord.json").read_text("utf-8"))),
        ("8B 直出（独立第三方：不同代际+不同规模）",
         json.loads((R / "_semi_qwen3vl_all.json").read_text("utf-8"))),
    ]
    for tag, ref in refs:
        analyse(tag, ref, q, cores, P, B)

    print("\n★ 重要限制（不要误读）")
    print("  · 只给**成分**，给不了成分各自的正确率——无 answer/gold ⇒ 不可观测")
    print("  · 同族参照（32B-AWQ）会退化成同义反复，故以 8B 为准")
    print("  · 「换模型之所以有效」的机制本身就可能是**误差去相关**")
    print("    ⇒ 若如此，共识过滤会把最有效的那部分改动滤掉")
    print("  ⇒ 要判断「该不该做共识过滤」，只能**用归零法去测**：")
    print("     造两个测量件——A 件把「共识集」退回基线、B 件把「孤例集」退回基线，")
    print("     两次得分之差即把 +101 条分解到两个子集上。")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
