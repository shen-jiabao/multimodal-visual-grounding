# -*- coding: utf-8 -*-
"""M5 测量件：读出「646 层中水平轴子类」当且仅当的错误条数 err_154。

★ 为什么不是原始 M5 模板
    模板要求字段 `gold` 与 `n_cands`，并要"抽 20 条**错误**样例"——三者在复赛都取不到：
      ① 复赛**无真值** ⇒ 没有 `gold`（这正是本项目用归零法的原因）
      ② `n_cands` 是"若跑一次枚举会得几个候选"，只有真跑了才知道；
         VLX 那次的 n_cands 是 **WeDetect 提案数**，与「Detect ALL instances of X」的枚举
         **不是同一个东西**，不能当 enum_ok_h
      ③ **归零法只给条数、不给身份** ⇒ 永远不知道哪几条错，"抽错误样例"做不到
    ⇒ 本件只测**一个数**：这 154 条里错了几条。

★ 协议（写死在提交之前，测完不得修改口径）
    件：把 154 条水平轴子类的框换成必然错的角落小框；其余全部保留真框
        （3941 + 646 里另外 492 + 序数 1103 = 5536 条）
    读数：
        C_rest  = round(得分 × 5690) − 689        ← 5536 条的正确数
        C_154   = 3730 − C_rest
        err_154 = 154 − C_154
        可行域：C_rest ≥ 3730 − 154 = 3576（即得分 ≥ 0.7499），低于此读数自相矛盾
    天花板换算：
        完美机制（修对全部 err_154、零破坏） ⇒ +err_154 条
        按枚举率 50%（SoM 实测的唯一参考）   ⇒ +err_154 × 0.5
    判据（沿用原案 10/20 阈值结构；movable_h ≈ err_154 × 枚举率，枚举率 ∈ [0.5, 1.0]）：
        err_154 < 20   ⇒ 不立 H
        20 ~ 40        ⇒ 边缘，立 H 则预登记更严（枚举率须先量）
        > 40           ⇒ 值得立 H
    本件**不测** enum_ok_h/movable_h —— 那需要真跑一次枚举（= H 的一半），不属本件范围。

用法
    python -B _measure_zero_axis154.py [--zip]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _sub_path as SP  # noqa: E402

BASE = "result_semi_n32_orddirect"
STEM = "result_semi_n32_orddirect__measure_zero_axis154"
N_ORD_CORRECT, N_NONORD_CORRECT = 689, 3730
N_SUP, N_AXIS = 646, 154          # ★154 为期望值，对不上就停（见 §核对）
DECOY_SIDE = 0.01
AXIS = re.compile(r"\b(leftmost|rightmost|topmost|rearmost)\b", re.I)
WORDS = ("leftmost", "rightmost", "topmost", "rearmost")


def decoy(i: int) -> list[float]:
    """必然错的角落小框；按索引在左上角微移，避免与任何真值重合。"""
    x = 0.001 + (i % 8) * 0.012
    y = 0.001 + (i // 8 % 8) * 0.012
    return [round(x, 6), round(y, 6), round(x + DECOY_SIDE, 6),
            round(y + DECOY_SIDE, 6)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    q = json.loads((R / "复赛数据集-基于大模型的多模态视觉理解与推理"
                    / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    sup = set(json.loads((R / "_vlx_sup_split.json").read_text(encoding="utf-8")))
    base = json.loads(SP.find(BASE).read_text(encoding="utf-8"))

    # ---- 核对：水平轴子类必须落在 646 层内，且条数须为 154 ----
    axis = sorted(k for k in sup if AXIS.search(q[k]["query"] or ""))
    print(f"646 层 {len(sup)} 条 → 命中水平轴词 {len(axis)} 条")
    if len(axis) != N_AXIS:
        comp = Counter(w for k in axis for w in WORDS
                       if re.search(r"\b" + w + r"\b", q[k]["query"], re.I))
        print(f"  ★条数 {len(axis)} ≠ 期望 {N_AXIS} ⇒ 按协议**停下查口径**，不出件")
        print(f"    成分：{dict(comp)}")
        return 2
    cnt = Counter(w for k in axis for w in WORDS
                  if re.search(r"\b" + w + r"\b", q[k]["query"], re.I))
    overlap = sum(cnt.values()) - N_AXIS
    print(f"  核对通过：{len(axis)} = {N_AXIS}")
    print(f"  分轴成分：{dict(cnt)}（和 {sum(cnt.values())}，"
          f"{'无重叠' if overlap == 0 else f'★有 {overlap} 条含两个轴词'}）")
    assert not (set(axis) & cores), "水平轴子类与序数层有交集"

    nonord = [k for k in base if k not in cores]
    zero, keep = axis, [k for k in nonord if k not in set(axis)]
    print(f"\n  非序数 {len(nonord)} = 零化 {len(zero)} + 保留 {len(keep)}；"
          f"序数 {len(cores)} 保留")
    assert len(keep) + len(zero) == len(nonord)

    out = {k: dict(v) for k, v in base.items()}
    for i, k in enumerate(zero):
        out[k]["bbox"] = decoy(i)

    moved_keep = [k for k in keep if out[k] != base[k]]
    moved_ord = [k for k in cores if out[k] != base[k]]
    still = [k for k in zero if out[k] == base[k]]
    assert not moved_keep, f"保留层被动过 {len(moved_keep)} 条"
    assert not moved_ord, f"序数层被动过 {len(moved_ord)} 条"
    assert not still, f"零化未生效 {len(still)} 条"
    ar = [round((out[k]["bbox"][2] - out[k]["bbox"][0])
                * (out[k]["bbox"][3] - out[k]["bbox"][1]), 8) for k in zero]
    assert all(0 < x <= 1e-4 for x in ar), f"诱饵面积异常 max={max(ar)}"
    leg = [k for k, v in out.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)} 条"
    bad = [k for k in q if any(out[k].get(f) != q[k][f]
                               for f in ("visible", "infrared", "depth", "query"))]
    assert not bad, f"字段回显异常 {len(bad)} 条"
    assert len(out) == 5690 and set(out) == set(q)
    print(f"  自检：保留 {len(keep)} 逐位未动 ✓｜序数 {len(cores)} 逐位未动 ✓｜"
          f"零化 {len(zero)} 全生效 ✓｜诱饵面积 ≤1e-4（max {max(ar):.1e}）✓")

    print(f"\n★ 读数协议（已写死）：")
    print(f"    C_rest  = round(得分×5690) − {N_ORD_CORRECT}")
    print(f"    C_154   = {N_NONORD_CORRECT} − C_rest")
    print(f"    err_154 = {N_AXIS} − C_154")
    print(f"    可行域：C_rest ≥ {N_NONORD_CORRECT - N_AXIS}（得分 ≥ "
          f"{(N_NONORD_CORRECT - N_AXIS + N_ORD_CORRECT)/5690:.4f}）")
    for p in (0.60, 0.748, 0.85):
        c = int(round(N_AXIS * p))
        print(f"    若子类 {p:.1%} ⇒ C_154={c:3d} ⇒ 本件得分 ≈ "
              f"{(N_NONORD_CORRECT - c + N_ORD_CORRECT)/5690:.4f}  ⇒ err_154={N_AXIS - c}")
    print(f"    判据：err_154 <20 不立 H｜20~40 边缘（枚举率须先量）｜>40 值得立 H")

    if a.zip:
        jp, zp = SP.out_path(STEM, ".json"), SP.out_path(STEM, ".zip")
        if jp.parent == SP.SUB:
            jp = SP.SUB / "未提交" / jp.name
            jp.parent.mkdir(exist_ok=True)
        if zp.parent == SP.SUB:
            zp = SP.SUB / "未提交" / zp.name
        jp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        payload = json.dumps(out, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo(f"{STEM}.json", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)
        print(f"\n已写 {jp}")
        print(f"     {zp}\n     SHA256 "
              f"{hashlib.sha256(zp.read_bytes()).hexdigest().upper()}")
        r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"),
                            str(zp)], capture_output=True, text=True)
        print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                                  if ln.strip().startswith(("[", "✓", "✗"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
