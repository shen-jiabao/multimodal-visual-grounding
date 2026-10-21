# -*- coding: utf-8 -*-
"""口径一致性核查：为什么 valid=37 而 ordered=144。

检查项：
  1. 纯 from-左右 的简单链（无残余词）是否被分组键拆分；
  2. count_ordered 与 count_A 的分组键差异；
  3. 三条真实链的成员打印。
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "复赛"))

from proxy_metrics import build_chain_groups, load_predictions  # noqa: E402
from src.inference.chains import chain_of, residual_tokens  # noqa: E402
from src.inference.matcher import parse_spatial_constraint  # noqa: E402

TAR = "outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz"


def count(key_mode):
    with open("dataset/queries/queries.json", encoding="utf-8") as fh:
        q = json.load(fh)
    pred = load_predictions(TAR)
    groups = defaultdict(list)
    for qid, v in q.items():
        b = pred.get(qid)
        if not b:
            continue
        cons = parse_spatial_constraint(v["query"])
        if cons is None or cons.get("kind") == "depth" or int(cons.get("rank", 0)) <= 0:
            continue
        if key_mode == "residual":
            key = (v["visible"], residual_tokens(v["query"]))
        else:
            ch = chain_of(v["query"])
            if ch is None:
                continue
            key = (v["visible"], ch["axis"], ch["direction"], ch["key"])
        groups[key].append((qid, v["query"], list(b), cons))
    ordered = a = 0
    multi = 0
    for key, items in groups.items():
        if len(items) < 2:
            continue
        multi += 1
        from src.inference.chains import classify_chain
        lab = classify_chain(items)
        if lab == "ordered":
            ordered += 1
        elif lab == "A":
            a += 1
    return len(groups), multi, ordered, a


def show_examples():
    with open("dataset/queries/queries.json", encoding="utf-8") as fh:
        q = json.load(fh)
    pred = load_predictions(TAR)
    groups = build_chain_groups(q, pred)
    shown = 0
    for key, items in groups.items():
        if len(items) < 2:
            continue
        print(f"\nKEY={key[1]} scene={os.path.basename(key[0])} n={len(items)}")
        for qid, query, box, cons in sorted(items, key=lambda x: x[3]["rank"]):
            print(f"   rank={cons['rank']} axis={cons['axis']} d={cons['direction']}"
                  f" box=[{box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f}]  {query}")
        shown += 1
        if shown >= 3:
            break


if __name__ == "__main__":
    for mode in ("residual", "chainkey"):
        n_groups, multi, ordered, a = count(mode)
        print(f"{mode:9s} 总链键={n_groups:5d} 多人链={multi:4d} ordered={ordered:4d} A={a:4d}")
    show_examples()
