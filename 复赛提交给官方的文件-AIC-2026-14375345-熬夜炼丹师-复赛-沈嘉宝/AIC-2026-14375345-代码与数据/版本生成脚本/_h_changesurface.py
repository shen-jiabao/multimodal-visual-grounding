# -*- coding: utf-8 -*-
"""H 的可观测口径：不测"错误组"（无真值时不可观测），改测**改动面**。

为什么换口径
    用户建议的 `movable_h = 错误组中 n_cands≥2 的条数` 需要知道"哪几条错"，
    但 queries.json 只有 ['depth','infrared','query','visible']，**没有 answer/gold**
    （复赛不提供任何真值，这正是本项目全程用分组测量法的原因）。
    ⇒ 错误组身份**原理上不可观测**，与样本量无关。

可观测的替代量
    改动面 = H 规则会选的框 与 **当前提交件（0.7786）的框** IoU < 0.5 的条数
      · IoU 高 ⇒ 规则与现状一致 ⇒ 不改（无收益、无风险）
      · IoU 低 ⇒ 会被改 ⇒ 进入改动面
    改动面 × 净成功率先验 = 可估收益；而改动面本身**不需要真值**。

规则（LTR 取最左 / RTL 取最右；topmost/rearmost 按 y 轴）
    leftmost  → cands[argmin x1]
    rightmost → cands[argmax x2]
    topmost   → cands[argmin y1]
    rearmost  → cands[argmax y2]

用法：python -B _h_changesurface.py [--som 路径] [--sub 路径]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
SUB = R / "submission_zips" / "semifinal" / "0.7786" / "result_semi_n32bf16_nonord.json"
AXIS = ("leftmost", "rightmost", "topmost", "rearmost")


def iou(a, b) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pick(word: str, cands: list) -> list | None:
    if not cands:
        return None
    w = word.lower()
    if w == "leftmost":
        return min(cands, key=lambda b: b[0])
    if w == "rightmost":
        return max(cands, key=lambda b: b[2])
    if w == "topmost":
        return min(cands, key=lambda b: b[1])
    if w == "rearmost":
        return max(cands, key=lambda b: b[3])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--som", default=str(R / "_axis154_som_partial.json"))
    ap.add_argument("--sub", default=str(SUB))
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    som = json.loads(Path(a.som).read_text(encoding="utf-8"))
    sub = json.loads(Path(a.sub).read_text(encoding="utf-8"))

    tot = len(som)
    ok = [k for k, v in som.items() if v.get("n_cands", 0) >= 2]
    print(f"枚举产物 {tot} 条（口径：应为 154，当前为部分）")
    print(f"  枚举成功（n_cands≥2）{len(ok)}/{tot} = {len(ok)/max(1,tot):.1%}")
    print(f"  枚举失败（n_cands≤1）{tot-len(ok)}/{tot} = {1-len(ok)/max(1,tot):.1%}")
    print(f"  参照：序数 SoM 枚举失败率 50.1%")

    print("\n=== 改动面（可观测口径）===")
    rows, words = [], {w: 0 for w in AXIS}
    for k in ok:
        rec = som[k]
        text = (q.get(k, {}).get("query") or "")
        hit = [w for w in AXIS if re.search(r"\b"+w+r"\b", text, re.I)]
        if not hit:
            continue
        w = hit[0]
        words[w] += 1
        exp = pick(w, rec.get("bbox_all") or [])
        cur = sub.get(k, {}).get("bbox")
        # 归一化到 0-1（若已是 0-1 则不变）
        mexp = [v/1000.0 for v in exp] if exp and max(exp) > 1.5 else exp
        mcur = [v/1000.0 for v in cur] if cur and max(cur) > 1.5 else cur
        rows.append((k, w, round(iou(mexp, mcur), 4)))

    rows.sort(key=lambda r: r[2])
    change = [r for r in rows if r[2] < 0.5]
    close = [r for r in rows if 0.5 <= r[2] < 0.9]
    same = [r for r in rows if r[2] >= 0.9]
    print(f"  可评估条目 {len(rows)}（分轴 {dict(words)}）")
    print(f"  ★改动面 IoU<0.5  : {len(change)}")
    print(f"    边界  0.5~0.9   : {len(close)}")
    print(f"    一致  ≥0.9      : {len(same)}")
    if rows:
        vals = sorted(r[2] for r in rows)
        print(f"  IoU p10={vals[len(vals)//10]:.3f} p50={vals[len(vals)//2]:.3f} "
              f"p90={vals[int(len(vals)*0.9)]:.3f}")
    print("\n  改动面前 15 条：")
    for k, w, io in change[:15]:
        print(f"    {k}  {w:9s} IoU={io:.3f}  {(q.get(k,{}).get('query') or '')[:58]}")

    print("\n=== 判据（跑前写定，分子换成改动面）===")
    n = len(change)
    if n < 10:      print(f"  改动面 {n} <10  ⇒ 不立 H")
    elif n <= 20:   print(f"  改动面 {n} 10~20 ⇒ 边缘（预先设定更严）")
    else:           print(f"  改动面 {n} >20  ⇒ 值得立 H")
    print("  注：本口径下 n 是**会被改的条数**（不是「能修好的条数」），")
    print("      故它是「暴露面」；收益仍须乘净成功率先验，且上限受 err_154=40 封顶。")
    print("\n  ★即使按最乐观 +20 条：(4430+20)/5690 ≈ 0.7821 ⇒ 离 0.8 仍差约 100 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())