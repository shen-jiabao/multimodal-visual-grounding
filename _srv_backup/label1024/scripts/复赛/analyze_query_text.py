from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from src.inference.matcher import parse_spatial_constraint

SPATIAL = [
    "leftmost", "rightmost", "behind", "beside", "above", "below", "near",
    "next to", "in front of", "closest", "farthest", "left of", "right of",
    "to the left", "to the right", "under", "on top of",
]
ORDINAL = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "from the left", "from the right",
]
ACTION = [
    "standing", "sitting", "walking", "running", "riding", "holding",
    "wearing", "facing", "taking", "flying", "parked", "moving", "lying",
    "squatting", "waving", "carrying", "pushing", "pulling", "climbing",
    "jumping", "sleeping", "eating", "drinking", "talking", "reading",
    "playing", "driving", "crossing", "waiting", "kneeling", "bending",
    "crouching", "looking", "watching", "pointing", "raising", "lifting",
    "opening", "closing", "entering", "exiting",
]


def has_any(q: str, words: list[str]) -> bool:
    ql = q.lower()
    return any(w in ql for w in words)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="queries.json 路径")
    ap.add_argument("--tokenizer", default=None, help="本地 tokenizer 目录（可选，缺省时跳过 token 统计）")
    ap.add_argument("--json", default=None, help="结果 JSON 输出路径")
    ap.add_argument("--max-token", type=int, default=250, help="同图合并分块 token 上限（默认 250）")
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as f:
        data = json.load(f)
    queries = {k: v["query"] for k, v in data.items()}
    total = len(queries)
    scenes = Counter(k.split("_")[0] for k in queries)
    n_scene = len(scenes)

    res: dict = {"total": total, "scenes": n_scene}

    comma_qs = {k: q for k, q in queries.items() if "," in q}
    res["comma"] = {
        "n": len(comma_qs),
        "tight_word": sum(1 for q in comma_qs.values() if re.search(r"\w,", q)),
        "comma_space": sum(1 for q in comma_qs.values() if re.search(r",\s", q)),
        "both": sum(1 for q in comma_qs.values() if re.search(r"\w,", q) and re.search(r",\s", q)),
    }
    wc = Counter()
    for q in comma_qs.values():
        for w in q.split():
            if w.endswith(","):
                wc[w.rstrip(",")] += 1
    res["comma"]["tight_words_unique"] = len(wc)
    res["comma"]["tight_words_top"] = wc.most_common(8)
    res["comma"]["overlap_spatial"] = sum(1 for q in comma_qs.values() if has_any(q, SPATIAL))
    res["comma"]["overlap_ordinal"] = sum(1 for q in comma_qs.values() if has_any(q, ORDINAL))
    res["comma"]["from_left_to_right"] = sum(1 for q in comma_qs.values() if "from left to right" in q.lower())

    apos_qs = {k: q for k, q in queries.items() if chr(39) in q}
    res["apostrophe"] = {
        "n": len(apos_qs),
        "possessive": sum(1 for q in apos_qs.values() if re.search(r"\w's\b", q)),
        "quote": sum(1 for q in apos_qs.values() if re.search(r"'\w+'", q)),
        "contraction": sum(1 for q in apos_qs.values() if re.search(r"\w(n't|'re|'ll|'ve|'d|'m)\b", q)),
    }

    paren_qs = {k: q for k, q in queries.items() if "(" in q or ")" in q}
    res["paren"] = {"n": len(paren_qs), "items": [(k, q) for k, q in paren_qs.items()]}

    cjk = re.compile(r"[\u4e00-\u9fff]")
    cjk_qs = {k: q for k, q in queries.items() if cjk.search(q)}
    cjk_lens = [
        len(m)
        for q in cjk_qs.values()
        for m in re.findall(r"'([^']*[\u4e00-\u9fff][^']*)'", q)
    ]
    res["cjk"] = {"n": len(cjk_qs), "quote_len": cjk_lens, "samples": [(k, q[:80]) for k, q in list(cjk_qs.items())[:10]]}

    res["nonascii"] = {"bom": sum(1 for q in queries.values() if "\ufeff" in q),
                       "u2011": sum(1 for q in queries.values() if "\u2011" in q),
                       "embedded_dot": sum(1 for q in queries.values() if re.search(r"\w\.\w", q))}

    res["semantic"] = {
        "spatial": sum(1 for q in queries.values() if has_any(q, SPATIAL)),
        "ordinal": sum(1 for q in queries.values() if has_any(q, ORDINAL)),
        "action": sum(1 for q in queries.values() if has_any(q, ACTION)),
    }
    cons_cnt = Counter()
    for q in queries.values():
        cons = parse_spatial_constraint(q)
        if cons:
            cons_cnt[cons.get("kind", "ordinal")] += 1
    res["constraint"] = dict(cons_cnt)

    if args.tokenizer and os.path.isdir(args.tokenizer):
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer)
            toks = [len(tok.encode(q, add_special_tokens=True)) for q in queries.values()]
            toks.sort()
            res["token"] = {
                "min": toks[0],
                "max": toks[-1],
                "mean": round(sum(toks) / len(toks), 2),
                "median": toks[len(toks) // 2],
                "p90": toks[int(len(toks) * 0.90)],
                "p99": toks[int(len(toks) * 0.99)],
                "over240": sum(1 for t in toks if t > 240),
            }
            merged = [
                sum(len(tok.encode(v["query"], add_special_tokens=True)) for k, v in data.items()
                    if k.split("_")[0] == sid)
                for sid in scenes
            ]
            res["token"]["scene_over_limit"] = sum(1 for t in merged if t > args.max_token)
            res["token"]["scene_max"] = max(merged)
        except Exception as e:
            res["token"] = {"error": f"{type(e).__name__}: {e}"}

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
