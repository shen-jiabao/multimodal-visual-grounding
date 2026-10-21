# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os

STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "with", "and", "to", "from", "at", "is",
    "are", "by", "for", "that", "this", "it", "its", "as", "or", "be", "was",
    "there", "which", "who", "his", "her", "their", "has", "have", "having",
    "not", "no", "all", "also", "than", "over", "under", "near", "beside",
}
SPATIAL_WORDS = {
    "left", "right", "top", "bottom", "above", "below", "front", "behind",
    "corner", "center", "middle", "side", "next", "far", "close", "edge",
    "between", "surrounding", "outermost", "innermost", "outer", "inner",
}
ORDINAL_WORDS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "leftmost",
    "rightmost", "nearest", "farthest", "closest", "furthest", "last",
}
COLOR_WORDS = {
    "white", "black", "red", "green", "blue", "yellow", "orange", "purple",
    "brown", "gray", "grey", "pink",
}


def tokenize(text: str) -> set[str]:
    return {t for t in text.lower().split() if t.isalpha() and t not in STOPWORDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="数据集根目录（相对项目根目录）")
    ap.add_argument("--json", default=None, help="导出统计 JSON 的路径（可选）")
    args = ap.parse_args()

    base = os.path.abspath(args.root)
    qpath = os.path.join(base, "queries", "queries.json")
    with open(qpath, encoding="utf-8") as f:
        q = json.load(f)

    groups: dict[str, list[tuple[str, str]]] = {}
    for qid, v in q.items():
        groups.setdefault(v["visible"], []).append((qid, v["query"]))

    stats: dict = {}
    n_multi = 0
    n_similar_pair_scene = 0
    n_shared_pair_scene = 0
    n_ordinal_scene = 0
    n_spatial_scene = 0
    n_verifiable_scene = 0
    total_pairs = 0
    pair_kinds = {"similar": 0, "shared": 0, "ordinal_both": 0, "spatial_both": 0,
                  "color_both": 0}
    scene_rows = []

    for vis, items in sorted(groups.items()):
        qids = [it[0] for it in items]
        queries = [it[1] for it in items]
        if len(queries) < 2:
            continue
        n_multi += 1
        toks = [tokenize(x) for x in queries]
        words = [t for t in toks]
        scene_has_similar = False
        scene_has_shared = False
        scene_has_ordinal = any(ORDINAL_WORDS & w for w in words)
        scene_has_spatial = any(SPATIAL_WORDS & w for w in words)
        for i in range(len(queries)):
            for j in range(i + 1, len(queries)):
                total_pairs += 1
                inter = words[i] & words[j]
                union = words[i] | words[j]
                if not union:
                    continue
                jaccard = len(inter) / len(union)
                if jaccard >= 0.4:
                    pair_kinds["similar"] += 1
                    scene_has_similar = True
                if len(inter) >= 2:
                    pair_kinds["shared"] += 1
                    scene_has_shared = True
                if (ORDINAL_WORDS & words[i]) and (ORDINAL_WORDS & words[j]):
                    pair_kinds["ordinal_both"] += 1
                if (SPATIAL_WORDS & words[i]) and (SPATIAL_WORDS & words[j]):
                    pair_kinds["spatial_both"] += 1
                if (COLOR_WORDS & words[i]) and (COLOR_WORDS & words[j]):
                    pair_kinds["color_both"] += 1
        if scene_has_similar:
            n_similar_pair_scene += 1
        if scene_has_shared:
            n_shared_pair_scene += 1
        if scene_has_ordinal:
            n_ordinal_scene += 1
        if scene_has_spatial:
            n_spatial_scene += 1
        if scene_has_similar or scene_has_shared or scene_has_ordinal or scene_has_spatial:
            n_verifiable_scene += 1
        scene_rows.append({
            "scene": os.path.basename(vis),
            "n_queries": len(queries),
            "similar": scene_has_similar,
            "shared": scene_has_shared,
            "ordinal": scene_has_ordinal,
            "spatial": scene_has_spatial,
            "verifiable": scene_has_similar or scene_has_shared or scene_has_ordinal or scene_has_spatial,
        })

    stats["n_multi_query_scenes"] = n_multi
    stats["total_query_pairs"] = total_pairs
    stats["pair_kinds"] = pair_kinds
    stats["scenes"] = {
        "similar_pair": n_similar_pair_scene,
        "shared_word_pair": n_shared_pair_scene,
        "any_ordinal": n_ordinal_scene,
        "any_spatial": n_spatial_scene,
        "verifiable": n_verifiable_scene,
    }
    stats["ratio_of_multi"] = {k: round(v / max(1, n_multi), 4)
                               for k, v in stats["scenes"].items()}
    stats["scene_examples"] = {
        "similar": [r for r in scene_rows if r["similar"]][:8],
        "ordinal": [r for r in scene_rows if r["ordinal"] and not r["similar"]][:8],
        "spatial_only": [r for r in scene_rows if r["spatial"] and not r["similar"] and not r["ordinal"]][:8],
    }

    print(f"多查询场景: {n_multi}")
    print(f"query 对总数: {total_pairs}")
    print("pair_kinds:", json.dumps(pair_kinds, ensure_ascii=False))
    print("scenes:", json.dumps(stats["scenes"], ensure_ascii=False))
    print("ratio_of_multi:", json.dumps(stats["ratio_of_multi"], ensure_ascii=False))

    vis_map = {v["visible"]: i for i, v in enumerate(q.values())}
    sample = list(groups.items())[:1]
    for vis, items in [("Images/visible/000001.png",
                        [("000001_001", "A squatting person"),
                         ("000001_002", "The white street light on the left side of the stone steps on the right"),
                         ("000001_003", "The second white patio umbrella in the outdoor seating area from left to right"),
                         ("000001_006", "Three people near the entrance of the commercial building")]),
                       ("Images/visible/019071.png",
                        [("019071_001", "The second tennis ball from the right to left"),
                         ("019071_002", "The third tennis ball from the right to left"),
                         ("019071_003", "The first tennis ball from the right to left")])]:
        toks = [tokenize(x) for _, x in items]
        print(f"\n示例场景 {vis}")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                inter = toks[i] & toks[j]
                union = toks[i] | toks[j]
                jac = len(inter) / len(union) if union else 0
                print(f"  {items[i][0]} × {items[j][0]}: jaccard={jac:.3f} "
                      f"shared={sorted(inter)}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=1)
        print(f"\n已导出: {args.json}")


if __name__ == "__main__":
    main()
