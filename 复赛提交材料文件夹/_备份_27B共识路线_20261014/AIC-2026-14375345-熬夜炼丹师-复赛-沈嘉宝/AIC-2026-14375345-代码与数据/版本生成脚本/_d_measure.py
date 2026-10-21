# -*- coding: utf-8 -*-
"""D 小样判定：改动率（判据①）与 IoU 分布（判据②）。

判据写定在 `_prereg_D.md` §5，本脚本**只做实现**，不含任何新判据：
    ① 改动率 r ≥ 0.465   （= 769 ÷ 1652；低于此到不了 0.8）
    ② 改动条目与基线的 IoU 不能大量落在 < 0.3（否则是"另一次独立猜测"）

用法
    python -B _d_measure.py <小样产物.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

BASE = "result_semi_n32_orddirect"     # 线上 0.7766（当前最高件）
THRESH_R = 0.465                       # 判据① 门槛
PRIOR = 0.173                          # 同机制边际成功率（两次独立验证）
N_SCOPE = 1652                         # D 作用范围全量条数


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("product")
    ap.add_argument("--sample", default=str(R / "_d_sample300.json"))
    a = ap.parse_args()

    sample = set(json.loads(Path(a.sample).read_text(encoding="utf-8")))
    base = json.loads(SP.find(BASE).read_text(encoding="utf-8"))
    d3 = json.loads(Path(a.product).read_text(encoding="utf-8"))

    print(f"基线 {BASE}（线上 0.7766）｜小样清单 {len(sample)} 条")
    print(f"三模态产物 {len(d3)} 条")
    assert set(d3) <= sample, \
        f"产物含清单外条目 {len(set(d3) - sample)} 条（越界）"
    missing = sample - set(d3)
    print(f"  清单内已出 {len(sample) - len(missing)}｜未出 {len(missing)}")

    st = Counter()
    changed = []
    for k in sorted(sample):
        rec = d3.get(k)
        if rec is None:
            st["0 未跑到"] += 1
            continue
        nb = rec.get("bbox")
        if not nb:
            st["1 无框 → 保留基线"] += 1
            continue
        nb = F.norm(nb)
        if not F.ordered(nb):
            st["2 框退化/逆序"] += 1
            continue
        if F.area(nb) > F.MAX_AREA:
            st["3 面积>0.35"] += 1
            continue
        if F.area(nb) < F.MIN_AREA:
            st["4 面积过小"] += 1
            continue
        io = F.iou(nb, base[k]["bbox"])
        if io >= F.IOU_KEEP:
            st["5 与基线一致(IoU≥0.5) → 不动"] += 1
            continue
        st["6 ★改动"] += 1
        changed.append((k, io, F.area(nb), F.area(base[k]["bbox"])))

    for k, v in sorted(st.items()):
        print(f"  {k:30s} {v}")
    tot = sum(st.values())
    assert tot == len(sample), f"分类合计 {tot} ≠ 样本 {len(sample)}（分类不互斥）"
    print(f"  合计 {tot} = 样本 {len(sample)} ✓")

    n = st["6 ★改动"]
    r = n / len(sample)
    print(f"\n★ 判据① 改动率 r = {n}/{len(sample)} = **{r:.3f}**"
          f"（门槛 {THRESH_R}）⇒ {'✅ 通过' if r >= THRESH_R else '❌ 未过'}")

    if changed:
        ios = sorted(io for _, io, _, _ in changed)
        lo = sum(1 for i in ios if i < 0.3)
        mid = sum(1 for i in ios if 0.3 <= i < 0.5)
        print(f"★ 判据② 改动条目与基线的 IoU：<0.3 的 {lo} 条（{lo/len(ios):.1%}）｜"
              f"0.3–0.5 的 {mid} 条｜p50 {ios[len(ios)//2]:.3f}")
        # 本项目对"另一次独立猜测"的经验判据：低重叠占多数即视为独立猜测
        bad = lo / len(ios) > 0.5
        print(f"   ⇒ {'❌ 低重叠占多数，先验不成立，停止' if bad else '✅ 未出现独立猜测特征'}")
        print(f"\n  样例（qid, IoU, 三模态面积, 基线面积）:")
        for k, io, ar, ab in changed[:8]:
            print(f"    {k}  IoU={io:.3f}  {ar:.4f} vs {ab:.4f}")

    exp = N_SCOPE * r * PRIOR
    print(f"\n若全量（{N_SCOPE} 条）维持此改动率：")
    print(f"  改动面 {N_SCOPE*r:.0f} × {PRIOR:.1%} ≈ **+{exp:.0f} 条** → "
          f"≈ {0.7766 + exp/5690:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())