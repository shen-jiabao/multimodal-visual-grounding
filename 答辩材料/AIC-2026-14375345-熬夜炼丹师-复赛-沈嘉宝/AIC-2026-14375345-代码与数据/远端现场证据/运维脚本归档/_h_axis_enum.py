# -*- coding: utf-8 -*-
"""H 第一步（修正版）：对 154 条**非序数轴类**查询做**枚举**，量枚举率与改动面。

为什么必须另写而不用管线
    `_qwen_full_pipeline.py` 的 Phase B 作用域硬编码为 `ordinal` 集合且不读 QWEN_QIDS：
        for k in by_img[imname]:
            if k in ordinal:            # ← 非序数条目永远进不来
                grp[cores[k].lower()].append(k)
    ⇒ 直接跑 Phase B 只会枚举序数条目（已实测：产物里 0 条轴类），对 H 无用。
    ⇒ 本脚本是**测量工具**（不改管线），只做枚举 + 几何选择。

★ 与序数 SoM 的关键结构差异（对 H 有利）
    序数题要模型「选第几个」（提示 → 枚举 → 叠编号 → 选中编号），引入"编号选错"失败源；
    轴类题的选是**几何确定的**（leftmost = x1 最小），**只需模型枚举**。
    故 H 的失败源只剩枚举一个。

★★ 核心词抽取规则（**跑之前写定，跑完不得改**）
    R1 去掉轴词本身（\b(leftmost|rightmost|topmost|rearmost)\b，大小写不敏感）
    R2 去掉开头的冠词（A / An / The）
    R3 在第一个介词处截断：of / on / in / at / with / beside / near / behind /
       under / above / below / next / by / from / for / to / stands / standing
       —— 取介词前的部分作核心词
    R4 若截断后为空，退化为核心词 = 去轴词去冠词后的整串
    R5 **参照物守卫**：若轴词前紧邻 `(of|beside|near|by|next to|in front of) the`
       等结构，说明轴词修饰的是**参照物**而非目标（实测样例：
       "The pale white flamingo ... to the right of the rightmost palm trunks"，
       答案是 flamingo、不是棕榈）⇒ 标记 REF，**不计入 H 的作用域**，单独统计

用法（远端执行）
    QWEN_API_MODEL=qwen3-vl-32b-bf16 python -B _h_axis_enum.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import defaultdict

os.environ.setdefault("QWEN_ENGINE", "api32b")
os.environ.setdefault("QWEN_API_MODEL", "qwen3-vl-32b-bf16")
os.environ.setdefault("QWEN_DATA", "/root/rivermind-data/dataset")

from PIL import Image  # noqa: E402

import _qwen_full_pipeline as P  # noqa: E402

BASE = "/root/rivermind-data"
AXIS = ("leftmost", "rightmost", "topmost", "rearmost")
AX_RE = re.compile(r"\b(" + "|".join(AXIS) + r")\b", re.I)
PREP = (r"of|on|in|at|with|beside|near|behind|under|above|below|next|by|from|"
        r"for|to|stands|standing|that|which|and")
REF_RE = re.compile(r"\b(of|beside|near|by|next\s+to|in\s+front\s+of)\s+the\s*$", re.I)


def extract_core(text: str) -> tuple[str, str]:
    """返回 (core, kind)；kind ∈ {OK, REF, EMPTY}。规则见模块 docstring R1~R5。"""
    t = text.strip().rstrip(".")
    m = AX_RE.search(t)
    if not m:
        return "", "EMPTY"
    # R5：轴词前紧邻参照物结构 ⇒ REF
    if REF_RE.search(t[:m.start()]):
        return "", "REF"
    # R1 去轴词
    t = (t[:m.start()] + " " + t[m.end():]).strip()
    # R2 去开头冠词
    t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.I)
    # R3 首个介词处截断
    mm = re.search(r"\s+(" + PREP + r")\b", t, re.I)
    if mm and mm.start() > 0:
        t = t[:mm.start()]
    # R4 退化
    t = re.sub(r"\s+", " ", t).strip(" ,;")
    if not t:
        return "", "EMPTY"
    return t, "OK"


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def geom_pick(word: str, cands: list) -> list | None:
    """轴类查询的选是几何确定的（不用模型）。"""
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
    ids = json.load(open(os.path.join(BASE, "_axis154.json"), encoding="utf-8"))
    q = json.load(open(os.path.join(P.DATA, "queries", "queries.json"), encoding="utf-8"))
    print(f"轴类 {len(ids)} 条｜model={P.API_MODEL}", flush=True)

    # 抽取核心词 + 分类
    info, refs, empt = {}, [], []
    for k in ids:
        text = q[k]["query"]
        w = AX_RE.search(text).group(1).lower() if AX_RE.search(text) else ""
        core, kind = extract_core(text)
        info[k] = {"word": w, "core": core, "kind": kind}
        if kind == "REF":
            refs.append(k)
        elif kind == "EMPTY":
            empt.append(k)
    print(f"  抽取：OK {sum(1 for v in info.values() if v['kind']=='OK')}"
          f"｜REF(参照物，排除) {len(refs)}｜EMPTY {len(empt)}", flush=True)
    for k in refs[:5]:
        print(f"    REF {k}  {q[k]['query'][:90]}")

    targets = [k for k in ids if info[k]["kind"] == "OK"]
    by_img = defaultdict(list)
    for k in targets:
        by_img[q[k]["visible"]].append(k)

    gen = P.make_gen_api()
    out, t0 = {}, time.time()
    for i, (imname, ks) in enumerate(sorted(by_img.items()), 1):
        try:
            img = Image.open(os.path.join(P.DATA, imname)).convert("RGB")
        except Exception as ex:
            for k in ks:
                out[k] = {"n_cands": 0, "reason": "img_open"}
            continue
        # 同一图内按 core 去重枚举（同 core 只枚举一次）
        per_core = defaultdict(list)
        for k in ks:
            per_core[info[k]["core"].lower()].append(k)
        for core, kk in per_core.items():
            try:
                etxt = gen(img, P.ENUM_PROMPT.format(core=core), max_new=512)
                cands = P.parse_boxes(etxt)
            except Exception as ex:
                cands = None
                print(f"[ERR] enum {imname}/{core}: {ex!r}", flush=True)
            n_c = len(cands) if cands else 0
            for k in kk:
                rec = {"core": core, "n_cands": n_c, "bbox_all": cands or []}
                if n_c <= 1:
                    rec["reason"] = "enum_fail"
                else:
                    rec["pick"] = geom_pick(info[k]["word"], cands)
                out[k] = rec
        if i % 10 == 0:
            print(f"    …{i}/{len(by_img)} 图（{len(out)} 条，{time.time()-t0:.0f}s）", flush=True)

    # 结算
    ok = [k for k, v in out.items() if v.get("n_cands", 0) >= 2]
    print(f"\n=== 枚举率 ===")
    print(f"  作用域（排除 REF/EMPTY 后）{len(targets)} 条")
    print(f"  ★枚举成功（n_cands≥2）{len(ok)}/{len(out)} = {len(ok)/max(1,len(out)):.1%}")
    print(f"    枚举失败 {len(out)-len(ok)}/{len(out)} = {1-len(ok)/max(1,len(out)):.1%}")
    print(f"    参照：序数 SoM 枚举失败率 50.1%")

    # 改动面（与当前提交件比）
    sub = json.load(open(os.path.join(BASE, "output", "_cur_sub.json"), encoding="utf-8"))
    chg, rows = 0, []
    for k in ok:
        cur = sub.get(k, {}).get("bbox")
        pk = out[k].get("pick")
        io = iou(pk, cur)
        rows.append((k, round(io, 4)))
    change = [r for r in rows if r[1] < 0.5]
    print(f"\n=== 改动面（与当前提交件 IoU<0.5）===")
    print(f"  ★改动面 {len(change)}（可评估 {len(rows)}）")
    if rows:
        v = sorted(r[1] for r in rows)
        print(f"  IoU p10={v[len(v)//10]:.3f} p50={v[len(v)//2]:.3f} p90={v[int(len(v)*0.9)]:.3f}")
    for k, io in change[:12]:
        print(f"    {k} {info[k]['word']:9s} IoU={io:.3f} core={info[k]['core'][:42]!r}")

    print(f"\n=== 判据（跑前写定，分子=改动面）===")
    n = len(change)
    tag = "不立 H" if n < 10 else ("边缘（预登记更严）" if n <= 20 else "值得立 H")
    print(f"  改动面 {n} ⇒ {tag}")
    print(f"  ★即使最乐观 +20：(4430+20)/5690 ≈ 0.7821 ⇒ 离 0.8 仍差约 100 条")

    json.dump({"info": info, "out": out, "refs": refs},
              open(os.path.join(BASE, "output", "axis154_enum2.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n已写 {BASE}/output/axis154_enum2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())