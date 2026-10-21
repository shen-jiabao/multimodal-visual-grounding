# -*- coding: utf-8 -*-
"""复赛 query 类型分类 + 三模态用法分析（决策支撑）。

目的：为"如何用三模态"提供**按 query 类型**的依据，而不是笼统地"三张图一起喂"。

分类口径
    一条 query 可命中多个标签；`primary`（主类型）按**优先级**取唯一一个，
    优先级顺序 = 该类型对模态的**特异性**（越依赖某一模态越靠前）：
        热/光照 > 距离/远近 > 序数/计数 > 空间关系 > 遮挡 > 动作 > 外观 > 类别
    这样主类型计数不会互相重叠，便于按类型定策略。

输出
    - 主类型分布（互斥，合计 = 总条数）
    - 标签分布（含重叠）
    - 每类的 序数/非序数 拆分（非序数 = 主战场）
    - 每类样例各 3 条，供目检判据是否成立

用法
    python -B _survey_modality_semi.py [--n 3]
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

import argparse
import json
import re
from collections import Counter
from pathlib import Path

R = Path(ROOT_S)
SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"

# 按优先级排列（前面的先判）。每条只落一个 primary。
TYPES = [
    ("热/光照（IR 特异）", re.compile(
        r"\b(dark|night|low.?light|dim|shadow|shade|hot|heat|cold|fire|flame|"
        r"warm|glow|glowing|lamp|lit|lighted|illuminat|bright|smoke|steam)\b",
        re.I)),
    ("距离/远近（Depth 特异）", re.compile(
        r"\b(closest|nearest|farthest|furthest|distance|far|near|faraway|"
        r"far away|foreground|background|depth|in front|in the back|"
        r"deepest|shallowest)\b", re.I)),
    ("序数/计数", re.compile(
        r"\b(first|second|third|fourth|fifth|last|leftmost|rightmost|topmost|"
        r"bottommost|frontmost|rearmost|only|single|sole|\d+(st|nd|rd|th))\b",
        re.I)),
    ("空间关系", re.compile(
        r"\b(left|right|top|bottom|upper|lower|above|below|under|over|beside|"
        r"next to|between|behind|north|south|east|west|corner|middle|center|"
        r"adjacent|alongside)\b", re.I)),
    ("遮挡/可见性", re.compile(
        r"\b(occlud|partial|partially|hidden|obscur|blocked|behind the|"
        r"emerging|peeking|visible|cut off)\b", re.I)),
    ("动作/行为", re.compile(
        r"\b(walking|running|sitting|standing|riding|carrying|holding|"
        r"talking|eating|drinking|laying|lying|crouch|kneel|jumping|"
        r"climbing|wearing|reading|using|playing|working|parked|moving)\b",
        re.I)),
    ("外观属性", re.compile(
        r"\b(red|blue|green|yellow|orange|purple|pink|brown|black|white|gray|"
        r"grey|silver|golden|metallic|wooden|plastic|leather|shiny|striped|"
        r"plaid|dotted|round|square|rectangular|tall|short|long|small|large|"
        r"thin|fat|old|new|young)\b", re.I)),
    ("类别/指代（兜底）", re.compile(r".", re.I)),
]

# 每类"主要指望的模态"（供映射表使用）
MODALITY = {
    "热/光照（IR 特异）": "IR 主 / RGB 辅",
    "距离/远近（Depth 特异）": "Depth 主 / RGB 辅",
    "序数/计数": "文本推算 + 框几何（Depth 仅作输入，禁止当排序键）",
    "空间关系": "框几何（x/y）+ RGB",
    "遮挡/可见性": "RGB + Depth（层次）",
    "动作/行为": "RGB（IR 仅作低光兜底）",
    "外观属性": "RGB（IR 仅作低光兜底）",
    "类别/指代（兜底）": "RGB",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="每类样例条数")
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads(
        (R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    print(f"总查询 {len(q)} 条｜序数 {len(cores)}｜非序数 {len(q)-len(cores)}"
          f"（★非序数 = 主战场：错题 940 条，占 69%）")

    primary = Counter()
    labels = Counter()
    nonord_in = Counter()
    samples: dict[str, list] = {t: [] for t, _ in TYPES}

    for k, v in q.items():
        t = v["query"]
        hit = [name for name, pat in TYPES if pat.search(t)]
        for h in hit:
            labels[h] += 1
        p = hit[0] if hit else TYPES[-1][0]
        primary[p] += 1
        if k not in cores:
            nonord_in[p] += 1
        if len(samples[p]) < a.n:
            samples[p].append(f"{k} | {t[:66]}")

    print(f"\n{'主类型':28s} {'条数':>6} {'占比':>7} {'其中非序数':>10}  主要指望的模态")
    print("-" * 96)
    for name, _ in TYPES:
        c = primary[name]
        print(f"{name:28s} {c:6d} {c/len(q):6.1%} {nonord_in[name]:10d}  "
              f"{MODALITY[name]}")
    print("-" * 96)
    tot = sum(primary.values())
    assert tot == len(q), f"主类型合计 {tot} ≠ 总数 {len(q)}（分类不互斥）"
    print(f"{'合计':28s} {tot:6d}   ✓ 互斥自检通过")

    print(f"\n标签分布（含重叠，一条可命中多个）：")
    for name, _ in TYPES[:-1]:
        c = labels[name]
        print(f"  {name:28s} {c:6d} {c/len(q):6.1%}")

    print(f"\n各类样例（供目检判据）：")
    for name, _ in TYPES:
        print(f"\n[{name}]  {MODALITY[name]}")
        for s in samples[name]:
            print("   ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())