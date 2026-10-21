# -*- coding: utf-8 -*-
"""零 GPU 分析：q38 的 +101 落在哪一层 + 序数层扩展的合法作用域 + 各方案预估。

三件事（全部离线，只用已有产物）
1. 519 条改动按「646 空间最高级 / 3941 其余」分层 —— 看 +101 从哪来。
2. 序数层（1103）里**当前框确为某个 direct 定位器输出**的条目数 —— 这就是
   「同机制」的合法作用域（B′ 的判据：IoU(当前框, direct 框) ≥ 0.999）。
   非 direct 来源（SoM / det 推算）属**跨机制**，按项目实测（5/5 负）不得纳入。
3. 输出各方案的量化预估（明确标注假设，不外推）。

用法
    python -B _q38_next_step.py
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
from pathlib import Path

R = Path(ROOT_S)
sys.path.insert(0, str(R))

import _sub_path as SP  # noqa: E402

BF16 = "result_semi_n32bf16_nonord"   # 0.7786 基线
Q38 = "result_semi_q38_nonord"        # 0.7963 当前最高
N = 5690
ORD_C = 689
NONORD_C = 3842                       # 0.7963 件的非序数桶正确数
SAME_MECH_PRIOR = (0.176, 0.161, 0.099, 0.194)   # 实测净成功率族
RATE_LO, RATE_HI = 0.10, 0.194        # 本件实测量与保守下限


def iou(a, b):
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def jload(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    cores = set(jload(R / "_ordinal_cores_r2.json"))
    sup = set(jload(R / "_vlx_sup_split.json"))
    bf16 = jload(SP.find(BF16))
    q38 = jload(SP.find(Q38))
    nonord = [k for k in bf16 if k not in cores]
    ch = [k for k in nonord if q38[k]["bbox"] != bf16[k]["bbox"]]
    print(f"非序数 {len(nonord)}｜改动 {len(ch)}")

    # ---------- 1. 改动面分层 ----------
    print("\n===== 1) 519 条改动的分层 =====")
    ch_sup = [k for k in ch if k in sup]
    ch_rest = [k for k in ch if k not in sup]
    print(f"  646 空间最高级层：改动 {len(ch_sup):>4}／该层 646  "
          f"= {len(ch_sup)/646:.1%}")
    print(f"  3941 其余层     ：改动 {len(ch_rest):>4}／该层 3941 "
          f"= {len(ch_rest)/3941:.1%}")
    print(f"  （q38 在非序数整体改动率 {len(ch)/len(nonord):.1%}）")
    print("  分层改动率之比 ⇒ 哪一层对'换模型'更敏感（不做因果推断，仅描述）")

    # ---------- 2. 序数层的合法作用域 ----------
    print("\n===== 2) 序数层扩展的合法作用域（同机制：当前框 = 某 direct 输出）=====")
    d32 = jload(R / "_qwen32b_direct_ord.json")          # 32B AWQ direct（全 5690）
    d8 = None
    for nm in ("_semi_qwen3vl_all.json",):
        p = R / nm
        if p.exists():
            d8 = jload(p)
            print(f"  8B direct 来源 {p.name}")
            break
    if d8 is None:
        print("  ★ 未找到 8B direct 产物（_semi_qwen3vl_all.json）⇒ 只能按 32B 判")

    n_8 = n_32 = n_both = n_none = 0
    scope = set()
    for k in sorted(cores):
        cur = q38[k]["bbox"]
        hit8 = bool(d8) and (d8.get(k) or {}).get("bbox") and \
            iou(cur, d8[k]["bbox"]) >= 0.999
        hit32 = (d32.get(k) or {}).get("bbox") and \
            iou(cur, d32[k]["bbox"]) >= 0.999
        if hit8 and hit32:
            n_both += 1
        elif hit8:
            n_8 += 1
        elif hit32:
            n_32 += 1
        else:
            n_none += 1
        if hit8 or hit32:
            scope.add(k)
    print(f"  当前框 == 8B direct  : {n_8}")
    print(f"  当前框 == 32B direct : {n_32}")
    print(f"  两者都等（同框）      : {n_both}")
    print(f"  **合法作用域（同机制）= {len(scope)}**")
    print(f"  其余 {n_none} 条来源为 SoM/det 推算 ⇒ 跨机制，不纳入")

    # ---------- 3. 各方案预估 ----------
    print("\n===== 3) 各方案量化预估（假设全部写明，不外推）=====")
    print(f"  现状：0.7963｜非序数桶正确 {NONORD_C}｜到 0.8 还差 "
          f"{(0.8-0.7963)*N:.1f} ≈ 21 条\n")

    def est(name, surface_lo, surface_hi, rate_lo, rate_hi, note):
        d_lo, d_hi = surface_lo * rate_lo, surface_hi * rate_hi
        s_lo, s_hi = 0.7963 + d_lo / N, 0.7963 + d_hi / N
        print(f"  {name}")
        print(f"    改动面 {surface_lo}~{surface_hi}｜净成功率 {rate_lo:.1%}~{rate_hi:.1%}"
              f"（本件实测 19.4%，下限取 10%）")
        print(f"    ⇒ 净 +{d_lo:.0f}~+{d_hi:.0f} 条 ⇒ 得分 {s_lo:.4f}~{s_hi:.4f}")
        print(f"    {note}\n")

    est("A. q38 扩到序数层（零下载，只占 GPU 时间）",
        int(len(scope) * 0.15), int(len(scope) * 0.25),
        RATE_LO, RATE_HI,
        f"作用域 {len(scope)}（实测）；改动率按非序数 11.3%~25%（B′ 实测 25.4%）取 15~25%")
    est("B. 加一个异质模型 InternVL3.5-38B-AWQ（推荐）",
        400, 700, RATE_LO, RATE_HI,
        "改动面按 q38 实测 519 的量级取 400~700（家族外模型，异质性>=q38）；"
        "需 GPU + 删盘腾 ~25G + 下载 ~21G")
    est("C. 4×40G 上 Qwen3-VL-235B-A22B-AWQ",
        500, 900, 0.13, RATE_HI,
        "改动面更大（模型更强、更不同）；需 4×40G + 删光现有模型 + 最长下载时间")
    print("  ⚠ 一切预估都受同一不确定性支配：**新模型在改动面上的净成功率**。")
    print(f"    本项目实测该量的四个值：{', '.join(f'{v:.1%}' for v in SAME_MECH_PRIOR)}")
    print("    上限不是 19.4%——那是 Qwen3.8 的实测，不能外推到别的模型。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
