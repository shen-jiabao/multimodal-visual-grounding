# -*- coding: utf-8 -*-
"""复赛 query 能力面分布（只读，不依赖模型/权重）。

回答的问题是"这个数据集到底在考什么"，从而判断哪一类改造对 ACC@0.5 有实际盘子。
分类互不互斥（一条 query 可同时是序数+部件），因此同时给出单类计数与联合计数。

用法：
    .venv\\Scripts\\python.exe scripts\\复赛\\diag_query_capability.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

# 本文件位于 scripts/复赛/ 下，项目根需上溯三级
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.inference.matcher import parse_spatial_constraint  # noqa: E402

# 能力面词表（与 scripts/复赛/analyze_query_text.py 的 SPATIAL/ORDINAL/ACTION 口径对齐并扩展）
ORDINAL = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
           "eighth", "ninth", "tenth", "leftmost", "rightmost", "topmost", "bottommost",
           "from the left", "from the right"]
SPATIAL = ["left", "right", "top", "bottom", "behind", "beside", "above", "below",
           "near", "next to", "in front of", "closest", "farthest", "furthest",
           "under", "on top of", "middle", "center", "background", "foreground",
           "between", "inside", "outside", "edge", "corner"]
THERMAL = ["warm", "warmest", "warmer", "cold", "coldest", "colder", "thermal",
           "heat", "glowing", "bright in the thermal", "temperature"]
DEPTH = ["closest", "nearest", "farthest", "furthest", "in the distance", "distant",
         "far away", "near the camera", "closest to the camera"]
PART = ["head", "face", "eye", "eyes", "hand", "hands", "arm", "leg", "legs", "foot",
        "feet", "shoe", "shoes", "sneaker", "ear", "nose", "mouth", "hair", "neck",
        "shoulder", "finger", "wrist", "knee", "thigh", "back", "chest", "tail",
        "wing", "beak", "paw", "hoof", "horn", "trunk of", "wheel", "wheels",
        "handle", "handles", "door", "window", "roof", "leg of", "seat", "wheel of"]
ACTION = ["standing", "sitting", "walking", "running", "riding", "holding", "wearing",
          "facing", "taking", "flying", "parked", "moving", "lying", "squatting",
          "waving", "carrying", "pushing", "pulling", "climbing", "jumping",
          "sleeping", "eating", "drinking", "talking", "reading", "playing",
          "driving", "crossing", "waiting", "kneeling", "bending", "crouching",
          "looking", "watching", "pointing", "raising", "lifting", "opening",
          "closing", "entering", "exiting", "leaning", "hanging", "dangling",
          "resting", "blooming", "walking on"]
COLOR = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
         "black", "white", "gray", "grey", "silver", "golden", "cyan", "beige"]
SUPERLATIVE = ["largest", "smallest", "biggest", "tallest", "shortest", "longest",
               "most", "least", "the only"]


def has_any(ql: str, words) -> bool:
    return any(w in ql for w in words)


def main() -> None:
    with open("dataset/queries/queries.json", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    cats = Counter()
    word_lens = []
    cjk = 0
    per_scene = Counter()
    cat_queries = {k: [] for k in
                   ("ordinal", "ordinal_parsed", "spatial", "depth", "thermal",
                    "part", "action", "color", "superlative", "cjk", "long")}

    for qid, v in data.items():
        q = v["query"]
        ql = q.lower()
        word_lens.append(len(ql.split()))
        per_scene[qid.split("_")[0]] += 1
        if re.search(r"[\u4e00-\u9fff]", q):
            cjk += 1
            cat_queries["cjk"].append((qid, q))
        if has_any(ql, ORDINAL):
            cats["ordinal"] += 1
            cat_queries["ordinal"].append((qid, q))
        if parse_spatial_constraint(q):
            cats["ordinal_parsed"] += 1
            cat_queries["ordinal_parsed"].append((qid, q))
        if has_any(ql, SPATIAL):
            cats["spatial"] += 1
        if has_any(ql, DEPTH):
            cats["depth"] += 1
            cat_queries["depth"].append((qid, q))
        if has_any(ql, THERMAL):
            cats["thermal"] += 1
            cat_queries["thermal"].append((qid, q))
        if has_any(ql, PART):
            cats["part"] += 1
            cat_queries["part"].append((qid, q))
        if has_any(ql, ACTION):
            cats["action"] += 1
        if has_any(ql, COLOR):
            cats["color"] += 1
        if has_any(ql, SUPERLATIVE):
            cats["superlative"] += 1
        if len(ql.split()) >= 20:
            cats["long_ge20_words"] += 1
            cat_queries["long"].append((qid, q))

    # 关系型 query：同时含方位词 + 另一个名词性锚点（"the X next to the Y" / "on the Z"）
    rel = 0
    rel_pat = re.compile(
        r"\b(on|in|at|under|behind|beside|near|next to|in front of|to the (left|right) of|"
        r"left of|right of|above|below|inside|between|with|worn by|held by|carried by|"
        r"ridden by|sitting on|standing on|attached to|mounted on|hanging (on|from))\b")
    for v in data.values():
        if rel_pat.search(v["query"].lower()):
            rel += 1

    # 联合：序数 + 部件（最容易互串的一类）
    both = sum(1 for v in data.values()
               if has_any(v["query"].lower(), ORDINAL) and has_any(v["query"].lower(), PART))

    print(f"复赛 query 总数: {n}")
    print(f"场景数: {len(per_scene)} | 每场景 query 数分布: "
          f"{dict(sorted(Counter(per_scene.values()).items()))}")
    wl = sorted(word_lens)
    print(f"词数: min={wl[0]} 中位={wl[len(wl)//2]} p90={wl[int(.9*len(wl))]} max={wl[-1]} "
          f"(>=20 词的 {cats['long_ge20_words']} 条)")
    print("\n--- 能力面命中数（百分比按 5690 计）---")
    for k in ("ordinal", "ordinal_parsed", "spatial", "depth", "thermal", "part",
              "action", "color", "superlative", "long_ge20_words"):
        print(f"  {k:18s} {cats[k]:5d}  {cats[k]/n*100:5.1f}%")
    print(f"  {'relational(锚点)':18s} {rel:5d}  {rel/n*100:5.1f}%")
    print(f"  {'cjk(中文)':18s} {cjk:5d}  {cjk/n*100:5.1f}%")
    print(f"  {'ordinal∩part':18s} {both:5d}  {both/n*100:5.1f}%")

    print("\n--- 各类样例（每类 3 条）---")
    for k, rows in cat_queries.items():
        if not rows:
            continue
        print(f"[{k}]")
        for qid, q in rows[:3]:
            print(f"   {qid}  {q[:110]}")


if __name__ == "__main__":
    main()
