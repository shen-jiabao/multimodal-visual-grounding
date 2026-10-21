# -*- coding: utf-8 -*-
"""离线量化：序数侧还有没有空间？——三闸门后的 SoM vs 32B 直接框。

动机
    实测显示冻结稿预设的序数 SoM 层天花板只有 120×25%≈+30（+0.005），据此
    得出"到 0.8 的主攻方向必须回到非序数层"。本脚本用**同一把尺子**去量序数侧
    的另一条路（32B 直接框，此前从未当杠杆用过），检验该结论是否成立。

必须先声明的三条限制（否则读数会被误用）
    1. 离线**没有真值**，这里量的是**改动面**（change surface），不是准确率。
       改动面是天花板的**上界**：净收益 = 改动面 × 净成功率。
    2. 25% 的同族先验来自历史上只动几十条的杠杆；对动几百条的情形应打折。
    3. 32B 直接框替换现行序数层属**跨机制**（现行序数层由枚举/SoM 派生）。
       按本项目实测边界"可迁移的不是规则文本，而是被替换的那一层的性质"，
       该先验能否迁移**未经验证**——所以本脚本只给上界，不给预测分。

用法
    python -B _ord_layer_levers.py
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
import sys
from collections import Counter
from pathlib import Path

R = Path(ROOT_S)
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
AXES = ("x", "y")
N = 5690


def load():
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8"))
    scope = set(json.loads((R / "_ordinal_cores_unres.json")
                           .read_text(encoding="utf-8")))
    base = json.loads(SP.find("result_semi_enum_trunc").read_text(encoding="utf-8"))
    som = json.loads((R / "_qwen32b_som_scoped.json").read_text(encoding="utf-8"))
    dord = json.loads((R / "_qwen32b_direct_ord.json").read_text(encoding="utf-8"))
    return q, cores, scope, base, som, dord


def bucket(nb, bb) -> str:
    """把候选框 nb 相对基线框 bb 归入互斥桶（与装配器同口径）。"""
    if not nb:
        return "1 无框 → 保留基线"
    nb = F.norm(nb)
    if not F.ordered(nb):
        return "2 框退化/逆序 → 保留基线"
    if F.area(nb) > F.MAX_AREA:
        return "3 面积>0.35 → 保留基线"
    if F.area(nb) < F.MIN_AREA:
        return "4 面积过小 → 保留基线"
    if F.iou(nb, bb) >= F.IOU_KEEP:
        return "5 与基线一致(IoU≥0.5) → 不动"
    return "6 ★改动"


def show(title: str, st: Counter, n_total: int) -> int:
    """打印桶分布，返回改动数。"""
    print(f"\n  {title}（{n_total} 条）")
    for k, v in sorted(st.items()):
        print(f"    {k:32s} {v:5d}")
    ch = st["6 ★改动"]
    print(f"    {'合计':32s} {sum(st.values()):5d}  "
          f"| ★改动面 {ch}（{ch / max(n_total, 1):.1%}）")
    return ch


def main() -> int:
    q, cores, scope, base, som, dord = load()
    ordinal = set(cores)
    print(f"基线 result_semi_enum_trunc | 序数 {len(ordinal)} | "
          f"不可推算作用域 {len(scope)}")
    assert scope <= ordinal, "作用域必须是序数的子集"

    # ---------- 路线 A：32B SoM（仅作用域），须过冻结稿三条闸门 ----------
    print("\n===== 路线 A：32B SoM（冻结稿指定机制）=====")
    g = Counter()
    stA = Counter()
    n_enum_fail = n_g1 = n_g2 = n_g3 = 0
    for k in sorted(scope):
        rec = som.get(k) or {}
        nb = rec.get("bbox")
        n = rec.get("n_cands") or 0
        if n <= 1:
            n_enum_fail += 1
        ax, _o = F.ordinal_axis(q[k]["query"], AXES)
        kk = F.ordinal_index(q[k]["query"])
        # 三条闸门各自独立计数（跑后实录 §1 的中间行靠这三行可复现）
        a1 = n >= 2
        a2 = bool(ax) and kk is not None and kk <= n
        a3 = bool(nb) and F.valid_cand(nb)
        n_g1 += a1
        n_g2 += a2
        n_g3 += a3
        if not (a1 and a2 and a3):
            g["未过闸门 → 保留原框"] += 1
            continue
        g["过闸门"] += 1
        b = bucket(nb, base[k]["bbox"])
        stA[b] += 1
    print(f"  枚举失败(n_cands≤1) {n_enum_fail} | 闸门1 候选≥2 {n_g1} | "
          f"闸门2 方向与k可抽 {n_g2} | 闸门3 框合法 {n_g3}")
    print(f"  三闸门：过 {g['过闸门']} | 未过 {g['未过闸门 → 保留原框']}  "
          f"（作用域 {len(scope)}）")
    chA = show("过闸门者的框分类", stA, g["过闸门"])

    # ---------- 路线 B：32B 直接框（全 1103 序数，跨机制）----------
    print("\n===== 路线 B：32B 直接框（跨机制，此前从未当杠杆用过）=====")
    stB = Counter()
    for k in sorted(ordinal):
        stB[bucket((dord.get(k) or {}).get("bbox"), base[k]["bbox"])] += 1
    chB = show("全部序数", stB, len(ordinal))

    stBs = Counter()
    for k in sorted(scope):
        stBs[bucket((dord.get(k) or {}).get("bbox"), base[k]["bbox"])] += 1
    chBs = show("其中：不可推算作用域 407", stBs, len(scope))

    stBr = Counter()
    for k in sorted(ordinal - scope):
        stBr[bucket((dord.get(k) or {}).get("bbox"), base[k]["bbox"])] += 1
    chBr = show("其中：可推算 696（冻结稿规定不动）", stBr, len(ordinal - scope))

    # ---------- 基线来源实证：407 与 696 不是同一来源 ----------
    # 这条是"32B 直接框该不该用"的判据：基线若是 8B 直接框 ⇒ 同族升级；
    # 基线若是 det 文本推算层 ⇒ 跨机制、先验不迁移。
    print("\n===== 基线来源：与 8B 直接框的吻合度 =====")
    d8b = json.loads((R / "_semi_qwen3vl_all.json").read_text(encoding="utf-8"))
    for nm, s in (("不可推算 407", scope), ("可推算 696", ordinal - scope)):
        eq = hi = lo = miss = 0
        for k in sorted(s):
            b8 = (d8b.get(k) or {}).get("bbox")
            if not b8:
                miss += 1
                continue
            io = F.iou(base[k]["bbox"], F.norm(b8))
            if io >= 0.999:
                eq += 1
            elif io >= 0.5:
                hi += 1
            else:
                lo += 1
        print(f"  {nm}：逐位相同 {eq} | IoU≥0.5 {hi} | IoU<0.5 {lo} | 8B无框 {miss}")

    # ---------- 路线 B′（纯同机制）：基线本身就是 8B 直接框 ----------
    # 这是与 17.6% 先验的**测量场景同类**的最小集合，判据为可检验条件：
    #   基线 == 8B 直接框（IoU ≥ 0.999） 且 32B 直接框给出不同且合法的框。
    # 依据：本项目已实测"可迁移的不是规则文本，而是**被替换那一层的性质**"；
    # 上一轮序数 SoM 层刚因跨机制（direct→SoM）而净 −7（约 6σ），故此处只收同机制项。
    print("\n===== 路线 B′（纯同机制：8B direct → 32B direct）=====")
    stP = Counter()
    for k in sorted(ordinal):
        b8 = (d8b.get(k) or {}).get("bbox")
        if not b8 or F.iou(base[k]["bbox"], F.norm(b8)) < 0.999:
            continue                     # 基线不是 8B 直接框 ⇒ 非同机制，直接排除
        nb = (dord.get(k) or {}).get("bbox")
        if not nb:
            stP["1 无框 → 保留基线"] += 1
            continue
        nb = F.norm(nb)
        if not F.valid_cand(nb):
            stP["2 框非法 → 保留基线"] += 1
            continue
        if F.iou(nb, base[k]["bbox"]) >= F.IOU_KEEP:
            stP["5 与基线一致(IoU≥0.5) → 不动"] += 1
            continue
        stP["6 ★改动"] += 1
    nP = show("基线 == 8B 直接框的序数条目（全 1103 中筛出）", stP,
              sum(stP.values()))
    nP_scope = 0
    for k in sorted(scope):
        b8 = (d8b.get(k) or {}).get("bbox")
        if not b8 or F.iou(base[k]["bbox"], F.norm(b8)) < 0.999:
            continue
        nb = (dord.get(k) or {}).get("bbox")
        if nb and F.valid_cand(F.norm(nb)) \
                and F.iou(F.norm(nb), base[k]["bbox"]) < F.IOU_KEEP:
            nP_scope += 1
    print(f"    其中落在不可推算作用域 407 内的: {nP_scope}")

    # ---------- 天花板对比（同一把尺子：改动面 × **实测**先验 17.6%）----------
    # 先验来源：32B 非序数件（467 改动 → 净 +82）= 17.6%，见跑后实录 §6.1。
    # ★该先验**只能用于同类机制**（direct→direct）；跨机制（direct→SoM）已实测 6σ 失败。
    PRIOR = 0.176
    print(f"\n===== 天花板对比（改动面 × 实测先验 {PRIOR:.1%} = 净翻转 → Δ得分）=====")
    print(f"  {'路线':34s} {'改动面':>7s} {'基线':>8s} {'Δ得分':>9s}  机制")
    BASE_NOW = 0.7742
    for nm, ch, mech in (
            ("A 32B SoM（冻结稿机制）", chA, "跨机制 ✗"),
            ("B′ 纯同机制（基线=8B direct）", nP, "同机制 ✓"),
            ("B 全部 1103 序数", chB, "混合"),
            ("B'' 可推算 696", chBr, "跨机制 ✗")):
        d = ch * PRIOR
        print(f"  {nm:34s} {ch:7d} {BASE_NOW:8.4f} {d / N:+9.4f}  {mech}"
              f"（→ {BASE_NOW + d / N:.4f}）")
    print(f"\n  基线 = 当前最高件 {BASE_NOW}（result_semi_n32_ordbase，线上实测）")
    print(f"  目标 0.8 → 缺口 {(0.8 - BASE_NOW) * N:.0f} 条；"
          f"非序数剩余错题 ≈ {940 - 82} 条")
    print("  ⚠ 先验只对**同类机制**可迁移（direct→direct）。跨机制项的上界不成立——"
          "序数 SoM 层实测 −10.6%（约 6σ），即此规则的代价。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())