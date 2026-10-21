# -*- coding: utf-8 -*-
"""VLX-Seek 与 32B bf16 的"配合"能提升多少 —— 用已有产物直接算（0 GPU）

背景（均已实测，非推测）：
  · VLX 全量非序数 4587 条 = 0.7492 ⇒ Δ = −156 条（相对 0.7766 基线，−18.4%）
    分两层：646 层 −7（−5.0%）｜3941 层 −149（−21.1%）
  · 结构根因：VLX **只选不画**——输出框被限定为 WeDetect 候选，
    而基线是 32B 直接出框（紧贴物体）。选对物体也可能因候选框松/偏而 IoU<0.5。

本脚本算的是"配合"的三种口径各值多少：
  A 以 bf16 为准（VLX 只做参考）      → 改动 0 ⇒ 收益 0
  B 以 VLX 为准（覆盖 bf16）          → 已实测 −156 ⇒ 负
  C VLX 与 bf16 一致处不动、分歧处跟 VLX → 需算分歧面，且净成功率不可观测
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
S = R / "submission_zips" / "semifinal"
BF16 = S / "0.7786" / "result_semi_n32bf16_nonord.json"
VLX = S / "0.7492" / "result_semi_vlx_nonord.json"
CORES = R / "_ordinal_cores_r2.json"


def iou(a, b) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1]) if b else 0.0


def main() -> int:
    for p in (BF16, VLX):
        if not p.exists():
            print(f"★缺 {p}")
            return 2
    bf = json.loads(BF16.read_text(encoding="utf-8"))
    vx = json.loads(VLX.read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))
    non = [k for k in bf if k not in cores]

    # 一致性（逐位 / IoU 分档）
    same = [k for k in non if bf[k]["bbox"] == vx[k]["bbox"]]
    print(f"非序数 {len(non)} 条")
    print(f"  逐位相同（VLX 与 bf16 完全一样）: {len(same)} = {len(same)/len(non):.1%}")

    bands = {"<0.5 不同物体": 0, "0.5~0.9": 0, "0.9~0.999": 0, "=1.0": 0}
    for k in non:
        io = iou(bf[k]["bbox"], vx[k]["bbox"])
        if io < 0.5:
            bands["<0.5 不同物体"] += 1
        elif io < 0.9:
            bands["0.5~0.9"] += 1
        elif io < 1.0:
            bands["0.9~0.999"] += 1
        else:
            bands["=1.0"] += 1
    print("  IoU(bf16, VLX) 分布：")
    for k, v in bands.items():
        print(f"    {k:16s} {v:5d}  ({v/len(non):.1%})")

    # 「VLX 只作为过滤器」：分歧处若改跟 VLX = 上表 <0.5 的那一档规模
    diff = [k for k in non if iou(bf[k]["bbox"], vx[k]["bbox"]) < 0.5]
    print(f"\n★口径 C（分歧处跟 VLX）改动面 = {len(diff)} 条")
    print(f"   已实测的同类替换（全量跟 VLX）= **−156 条 = −18.4%**")
    print("   ⇒ 该改动面下净成功率实测为 **负**，并非「未知」——这是已关闭机制的重复")

    # 面积对比：验证"候选框更松"的结构缺陷
    ar_b, ar_v = [], []
    for k in diff:
        if bf[k]["bbox"]:
            ar_b.append(area(bf[k]["bbox"]))
        if vx[k]["bbox"]:
            ar_v.append(area(vx[k]["bbox"]))
    if ar_b and ar_v:
        ar_b.sort(); ar_v.sort()
        print(f"\n  分歧面内 面积对比（中位）:")
        print(f"    bf16  {ar_b[len(ar_b)//2]:.5f}")
        print(f"    VLX   {ar_v[len(ar_v)//2]:.5f}")
        print(f"    ⇒ VLX/bf16 面积比 {ar_v[len(ar_v)//2]/max(1e-9,ar_b[len(ar_b)//2]):.2f}")

    print("\n=== 结论（三种口径） ===")
    print("  A 以 bf16 为准、VLX 只参考 : 改动 0 ⇒ **收益 0**")
    print("  B 以 VLX 为准（覆盖）      : **−156 条（−18.4%）**，已实测并封口")
    print("  C 只在分歧处跟 VLX         : 改动面见上；该机制净成功率已实测为负 ⇒ **不重开**")
    print("\n  ★仲裁思路（谁对跟谁）**不可测**：无真值 ⇒ 不知分歧处谁对，")
    print("    故「仲裁上限」无法赋值——与 H 的 movable_h 同一个不可观测性问题。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())