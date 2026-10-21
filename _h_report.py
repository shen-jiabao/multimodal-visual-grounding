# -*- coding: utf-8 -*-
"""H 测量结果 → 用户要的四行读数（本地，0 GPU）

四行：
  ① 作用域条数（排除 REF/EMPTY 后）
  ② 全部枚举率（n_cands≥2）              ← 与序数 SoM 50.1% 可比
  ③ 几何选择唯一率 / 平局率              ← 按 _prereg_H.md §5 主口径，ε=0.02
  ④ 改动面（强口径 IoU=1.0 之外 / 弱口径 IoU<0.5）  ← 判据分子（强）

规则全部取自 _prereg_H.md（跑之前写定）：
  主口径 leftmost/rightmost 用 x1；topmost/rearmost 用 y1；ε=0.02
  备选口径 cx / x2 / y2 —— 只作敏感性对照，不用于判据

用法：python -B _h_report.py [--res 远端产物路径] [--sub 当前件路径]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal" / "0.7786" / "result_semi_n32bf16_nonord.json"
RES = R / "_axis154_enum2.json"
EPS = 0.02                      # 平局容差：沿用项目既有序数容差（_prereg_H §5）


def iou(a, b) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def key_main(word: str, b: list) -> float:
    return b[0] if word in ("leftmost", "rightmost") else b[1]


def key_alt(word: str, b: list) -> float:
    if word in ("leftmost", "rightmost"):
        return (b[0] + b[2]) / 2.0          # cx
    return (b[1] + b[3]) / 2.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(RES))
    ap.add_argument("--sub", default=str(SUB))
    a = ap.parse_args()

    d = json.loads(Path(a.res).read_text(encoding="utf-8"))
    info, out, refs = d["info"], d["out"], d.get("refs", [])
    sub = json.loads(Path(a.sub).read_text(encoding="utf-8"))

    ok_ids = [k for k, v in info.items() if v["kind"] == "OK"]
    ref_ids = [k for k, v in info.items() if v["kind"] == "REF"]
    emp_ids = [k for k, v in info.items() if v["kind"] == "EMPTY"]

    print("=" * 66)
    print(f"① 作用域条数（排除 REF/EMPTY 后）: {len(ok_ids)}"
          f"　（REF {len(ref_ids)}｜EMPTY {len(emp_ids)}｜原始 {len(info)}）")
    print(f"   REF 明细：")
    for k in ref_ids[:6]:
        print(f"     {k}  {(info[k].get('word') or '')}  core={info[k].get('core')!r}")

    enum_ok = [k for k in ok_ids if out.get(k, {}).get("n_cands", 0) >= 2]
    print("\n" + "=" * 66)
    print(f"② 全部枚举率（n_cands≥2）: {len(enum_ok)}/{len(ok_ids)} = "
          f"{len(enum_ok)/max(1,len(ok_ids)):.1%}")
    print(f"   枚举失败: {len(ok_ids)-len(enum_ok)}/{len(ok_ids)} = "
          f"{1-len(enum_ok)/max(1,len(ok_ids)):.1%}　（序数 SoM 失败率 50.1%）")

    # ③ 几何选择唯一率 / 平局率
    tie, uniq, nofit = [], [], []
    for k in enum_ok:
        cands = out[k].get("bbox_all") or []
        w = info[k]["word"]
        if not cands or not w:
            nofit.append(k)
            continue
        vals = [key_main(w, b) for b in cands]
        best = min(vals) if w in ("leftmost", "topmost") else max(vals)
        near = [i for i, v in enumerate(vals) if abs(v - best) < EPS]
        (tie if len(near) > 1 else uniq).append(k)
    tot_geo = len(uniq) + len(tie)
    print("\n" + "=" * 66)
    print(f"③ 几何选择（主口径 x1/y1，ε={EPS}）")
    print(f"   ★唯一 {len(uniq)}/{tot_geo} = {len(uniq)/max(1,tot_geo):.1%}"
          f"　★平局 {len(tie)}/{tot_geo} = {len(tie)/max(1,tot_geo):.1%}")
    if len(uniq) + len(tie) < len(enum_ok):
        print(f"   （{len(nofit)} 条无轴词或无候选，未参与）")

    # 备选口径差异
    diff = 0
    for k in enum_ok:
        cands = out[k].get("bbox_all") or []
        w = info[k]["word"]
        if not cands or not w:
            continue
        vm = [key_main(w, b) for b in cands]
        va = [key_alt(w, b) for b in cands]
        bm = min(vm) if w in ("leftmost", "topmost") else max(vm)
        ba = min(va) if w in ("leftmost", "topmost") else max(va)
        im = vm.index(bm)
        ia = va.index(ba)
        if im != ia:
            diff += 1
    print(f"   备选口径（cx/y 中心）与主口径**给出不同候选**的条数：{diff}"
          f"　（差异大 ⇒ 几何选择本身不唯一）")

    # ④ 改动面（仅在 唯一 且 枚举成功 的替换集上算）
    # 强口径 IoU < 0.999（噪声带 0.999~1.0 单列，不计入）——见 _prereg_H §6
    repl = uniq
    strong, weak, noise, rows = 0, 0, 0, []
    for k in repl:
        cands = out[k].get("bbox_all") or []
        w = info[k]["word"]
        vals = [key_main(w, b) for b in cands]
        best = min(vals) if w in ("leftmost", "topmost") else max(vals)
        pick = cands[vals.index(best)]
        cur = sub.get(k, {}).get("bbox")
        io = iou(pick, cur)
        rows.append((k, round(io, 6)))
        if io < 0.999:
            strong += 1
        elif io < 1.0:
            noise += 1
        if io < 0.5:
            weak += 1
    rows.sort(key=lambda r: r[1])
    print("\n" + "=" * 66)
    print(f"④ 改动面（替换集 = 枚举成功且非平局，共 {len(repl)} 条）")
    print(f"   ★强口径（IoU < 0.999）      : {strong}　← 判据分子")
    print(f"    噪声带（0.999 ≤ IoU < 1.0）: {noise}　← 浮点末位，不计入")
    print(f"    弱口径（IoU < 0.5）        : {weak}")
    print(f"    与当前件完全一致（IoU = 1.0）: {len(repl)-strong-noise}")

    # ★带宽分布：区分「换了实例」与「同一物体的框刷新」
    bands = {"[0, 0.5) 换实例": 0, "[0.5, 0.9) 框偏移": 0,
             "[0.9, 0.999) 框微偏": 0, "[0.999, 1.0] 噪声带": 0}
    for _, io in rows:
        if io < 0.5:
            bands["[0, 0.5) 换实例"] += 1
        elif io < 0.9:
            bands["[0.5, 0.9) 框偏移"] += 1
        elif io < 0.999:
            bands["[0.9, 0.999) 框微偏"] += 1
        else:
            bands["[0.999, 1.0] 噪声带"] += 1
    print("\n   ★IoU 带宽分布（区分「换实例」与「同物框刷新」）：")
    for k, v in bands.items():
        print(f"      {k:22s} {v:3d}  ({v/max(1,len(repl)):.1%})")
    print("   读法：只有「换实例」那一档是 H 的**真正用武之地**（选错实例 ⇒ 有机会修对）；")
    print("         其余档是**同一物体的框刷新** ⇒ 无上升空间、只有把 IoU 推到 0.5 以下的风险。")

    # 前置门槛（_prereg_H §7，跑前拍定的 20%）
    sen_pct = diff / max(1, len(enum_ok))
    tie_pct = len(tie) / max(1, tot_geo)
    print("\n" + "=" * 66)
    print("前置门槛（_prereg_H §7，阈值 20% 为跑前拍定）")
    print(f"  敏感性（两口径不同候选）{diff}/{len(enum_ok)} = {sen_pct:.1%}"
          f"　{'★越界' if sen_pct > 0.20 else '通过'}")
    print(f"  平局率（tie）          {len(tie)}/{tot_geo} = {tie_pct:.1%}"
          f"　{'★越界' if tie_pct > 0.20 else '通过'}")

    n = strong
    if sen_pct > 0.20:
        verdict = "不立 H（前置门槛：敏感性越界）"
    elif tie_pct > 0.20:
        verdict = "不立 H（前置门槛：平局率越界）"
    elif n < 10:
        verdict = "不立 H（收益上限都不够）"
    elif n <= 20:
        verdict = "边缘（上限刚够，净成功率未知，风险大）"
    else:
        verdict = "值得立 H（上限较高、成本低）"
    print("\n" + "=" * 66)
    print(f"判定（_prereg_H §7，**只用改动面作上限判据**）：{verdict}（强口径 {n}）")
    print("  净成功率**不可观测**（无真值）⇒ 预登记已声明「可能为负」，不给期望收益")
    print("  天花板提醒：即使改动面 29，(4430+29)/5690 ≈ 0.7837 ⇒ 离 0.8 仍差约 100 条")
    print("  错误组枚举率 / movable_h：**不可测**（queries.json 无 answer）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())