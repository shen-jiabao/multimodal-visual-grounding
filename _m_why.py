#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M 为何低分：把还能离线算的证据全算出来。

口径：拿**最终提交件**（result_semi_m3m_rules5_mb）与基线（0.7963）逐位比，
"不同的那批"就是装配器真正采纳的改动（= 1330 条），对它们做几何剖面。

可回答：
  ① 全部/分层 的 改动条数 与 IoU 剖面
  ② M 的框相对基线是"更紧"还是"更松"（面积比）——检验"框紧"那条规则是否是元凶
  ③ 序数层 vs 非序数层 的破坏是否不均
  ④ M 的"整体失效"型输出（退化/超面积/整幅）规模
不可回答（明确写出）：−298 在序数/非序数之间怎么分——平台只给总分。
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal"
M = SUB / "未提交" / "result_semi_m3m_rules5_mb.json"
BEST = SUB / "0.7963" / "result_semi_q38_nonord.json"
RAW = R / "_m_direct_filled.json"
CORES = R / "_ordinal_cores_r2.json"

MAX_AREA, MIN_AREA = 0.35, 1e-5


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


def q(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)] if xs else float("nan")


def main() -> int:
    m = json.loads(M.read_text(encoding="utf-8"))
    b = json.loads(BEST.read_text(encoding="utf-8"))
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))

    chg = [k for k in m if m[k]["bbox"] != b[k]["bbox"]]
    print(f"最终件 vs 基线：改动 {len(chg)} 条")
    ordc = [k for k in chg if k in cores]
    non = [k for k in chg if k not in cores]
    print(f"  序数 {len(ordc)}（占序数层 {len(ordc)/len(cores)*100:.1f}%）"
          f"｜非序数 {len(non)}（占非序数层 "
          f"{len(non)/(len(m)-len(cores))*100:.1f}%）")

    print("\n=== ① 改动条的 IoU 剖面（与基线）===")
    ious = [iou(m[k]["bbox"], b[k]["bbox"]) for k in chg]
    for lo, hi, nm in ((0.0, 0.3, "[0,0.3) 区域几乎无关"),
                       (0.3, 0.5, "[0.3,0.5) 部分重叠"),
                       (0.5, 1.01, "[0.5,1] 不该出现（守卫挡住了）")):
        c = sum(1 for x in ious if lo <= x < hi)
        print(f"  {nm:<28} {c:>5} ({c/max(1,len(ious))*100:5.1f}%)")
    print(f"  中位 IoU {st.median(ious):.4f}")
    print("  ★注意：装配器 IOU_KEEP=0.5 ⇒ **所有** 被采纳的改动都满足 IoU<0.5，"
          "\n     即 M 与基线在这些位置**判到了不同区域**。")

    print("\n=== ② 框紧/框松：改动条的面积比 M/基线 ===")
    for nm, ks in (("全部改动", chg), ("序数", ordc), ("非序数", non)):
        if not ks:
            continue
        r = [area(m[k]["bbox"]) / max(area(b[k]["bbox"]), 1e-9) for k in ks]
        am = [area(m[k]["bbox"]) for k in ks]
        ab = [area(b[k]["bbox"]) for k in ks]
        shrink = sum(1 for x in r if x < 0.8)
        grow = sum(1 for x in r if x > 1.25)
        print(f"  {nm:<8} n={len(ks):>4}｜面积比 p10 {q(r,.1):.2f} "
              f"p50 {q(r,.5):.2f} p90 {q(r,.9):.2f}"
              f"｜M 明显更小 {shrink}（{shrink/len(ks)*100:.0f}%）"
              f"／更大 {grow}（{grow/len(ks)*100:.0f}%）")
        print(f"           面积中位  M {q(am,.5):.4f} vs 基线 {q(ab,.5):.4f}")

    print("\n=== ③ M 原始输出的整体失效规模（全体 5690）===")
    A = [area(raw[k]["bbox"]) for k in raw if raw[k].get("bbox")]
    deg = sum(1 for k in raw if raw[k].get("bbox")
              and not (raw[k]["bbox"][0] < raw[k]["bbox"][2]
                       and raw[k]["bbox"][1] < raw[k]["bbox"][3]))
    print(f"  有框 {len(A)}｜面积>0.35 的 {sum(1 for a in A if a > MAX_AREA)}"
          f"｜面积<1e-5 的 {sum(1 for a in A if a < MIN_AREA)}"
          f"｜整幅(≥0.99) 的 {sum(1 for a in A if a >= 0.99)}"
          f"｜框退化/逆序 {deg}")
    B = [area(b[k]["bbox"]) for k in b if b[k].get("bbox")]
    print(f"  基线同项：     面积>0.35 的 {sum(1 for a in B if a > MAX_AREA)}"
          f"｜整幅(≥0.99) 的 {sum(1 for a in B if a >= 0.99)}")
    print(f"  面积中位：M {q(A,.5):.4f}  vs  基线 {q(B,.5):.4f}")

    print("\n=== ④ 两侧都被守卫拒掉/保留的规模（M 的输出有多少根本没被采纳）===")
    keep = len(m) - len(chg)
    print(f"  最终件里沿用基线的 {keep} 条（{keep/len(m)*100:.1f}%）"
          f"｜采纳 M 的 {len(chg)} 条")
    print("  ⇒ M 即便在'与基线一致'的 4186 条上更好，也**不会被计入**；"
          "\n     反之亦然。故 −298 只是'分歧面'上的净账。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
