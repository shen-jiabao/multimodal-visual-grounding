# -*- coding: utf-8 -*-
"""Qwen3.8-27B 件的**事前区间估计**（不是判据，判据在 `_prereg_I.md` §4/§6）。

方法：用项目已验证过的诊断量 —— **改动面与基线的 IoU 分布**。
    B′（序数 87 改动，IoU 分布健康）→ 净 +16.1%  ⇒ 先验适用
    D （128 改动，p50 = 0.000、<0.3 占 80.5%）→ 净 −3.1% ⇒ 先验不适用

同口径对照：bf16 件（同族精度修复，111 改动，实测净 +9.9%）的改动面 IoU 分布。
    · 若本件分布与 bf16 件相近 ⇒ 支持"同机制先验可迁移"
    · 若本件分布退化成 D 那样（大量接近 0）⇒ 支持"跨代无先验、近似重掷"

★ 声明：本脚本输出**只用于回答"预计得分"**，不参与任何闸门，也不改变 §6 的止损线。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _sub_path as SP  # noqa: E402

CORES = R / "_ordinal_cores_r2.json"
BF16 = "result_semi_n32bf16_nonord"      # 0.7786 基线（bf16 件）
ORDD = "result_semi_n32_orddirect"       # 0.7766（bf16 件的基线）
Q38 = "result_semi_q38_nonord"           # 本件


def iou(a, b):
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dist(name, pairs):
    """pairs = [(qid, 与基线 IoU)]；打印分布。"""
    vs = sorted(v for _, v in pairs)
    if not vs:
        print(f"  {name}: 空")
        return None
    q = lambda t: vs[min(int(len(vs) * t), len(vs) - 1)]
    lo = sum(1 for v in vs if v < 0.3) / len(vs)
    print(f"  {name}: n={len(vs)}  p10={q(.1):.3f} p50={q(.5):.3f} p90={q(.9):.3f}"
          f"  IoU<0.3 占 {lo:.1%}")
    return {"n": len(vs), "p50": q(.5), "lt03": lo}


def main() -> int:
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))
    bf16 = json.loads(SP.find(BF16).read_text(encoding="utf-8"))
    ordd = json.loads(SP.find(ORDD).read_text(encoding="utf-8"))
    q38 = json.loads(SP.find(Q38).read_text(encoding="utf-8"))
    nonord = [k for k in bf16 if k not in cores]
    print(f"非序数 {len(nonord)}")

    # bf16 件的改动面：与它的基线（0.7766 件）逐位不同的非序数条目
    ch_bf16 = [(k, iou(bf16[k]["bbox"], ordd[k]["bbox"]))
               for k in nonord if bf16[k]["bbox"] != ordd[k]["bbox"]]
    # 本件改动面：与它的基线（0.7786 件）逐位不同
    ch_q38 = [(k, iou(q38[k]["bbox"], bf16[k]["bbox"]))
              for k in nonord if q38[k]["bbox"] != bf16[k]["bbox"]]

    print("\n=== 改动面与基线的 IoU 分布（同口径）===")
    a = dist(f"bf16 件 (实测净 +9.9%)", ch_bf16)
    b = dist(f"本件 q38   (待测)", ch_q38)
    # 项目里的两个锚（供对照，非本件数据）
    print("  对照锚：B′ 净 +16.1%（分布健康）｜D 净 −3.1%（IoU<0.3 占 80.5%）")

    print("\n=== 区间估计（改动面 × 净成功率）===")
    n = len(ch_q38)
    base = 0.7786
    for label, rate in (("先验适用（=同机制常数 17.3%）", 0.173),
                        ("折半（跨代打对折 8.7%）", 0.087),
                        ("近似重掷（D 型 −3.1%）", -0.031),
                        ("掷硬币（0%）", 0.0)):
        d = n * rate
        print(f"  {label:28s} 净 {d:+7.1f} 条 → 得分 ≈ {base + d / 5690:.4f}")
    print(f"\n  注：读数口径 得分×5690 − 689 对 3741（基线非序数桶正确数）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
