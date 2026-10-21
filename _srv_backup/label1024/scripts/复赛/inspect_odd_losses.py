# -*- coding: utf-8 -*-
"""打印既非截断、也非 echo_fail 的少数丢框记录（逐条人工可判）。

上一轮 diag 归因：92 条丢框里 89 条是 truncated_json/truncated_budget，
剩余 1 条 no_pairs + 2 条 parse_or_invalid 不属于截断故事，需单独看。
"""
from __future__ import annotations

import argparse
import json

TAIL_RATIO = 0.95


def classify(rec, mnt):
    lens = rec.get("reply_lens") or []
    pairs = rec.get("parsed_pairs") or []
    unmatched = rec.get("unmatched_labels") or []
    brace = rec.get("brace_unclosed") or []
    if rec.get("matched") is True:
        return "matched"
    if not lens:
        return "no_reply_record"
    if any(int(b) > 0 for b in brace):
        return "truncated_json"
    if unmatched:
        return "echo_fail"
    if max(lens) >= mnt * TAIL_RATIO:
        return "truncated_budget"
    if pairs:
        return "parse_or_invalid"
    return "no_pairs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--kinds", nargs="+", default=["no_pairs", "parse_or_invalid"])
    args = ap.parse_args()

    shown = 0
    with open(args.diag, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = classify(rec, args.max_new_tokens)
            if kind not in args.kinds:
                continue
            shown += 1
            print(f"--- [{kind}] {rec['query_id']}  {rec['visible'].split('/')[-1]}")
            print(f"    query      : {rec['query']}")
            print(f"    chunk      : #{rec.get('chunk_index')} size={rec.get('chunk_size')}")
            print(f"    reply_lens : {rec.get('reply_lens')}")
            print(f"    brace      : {rec.get('brace_unclosed')}")
            print(f"    pairs      : {json.dumps(rec.get('parsed_pairs'), ensure_ascii=False)}")
            print(f"    unmatched  : {rec.get('unmatched_labels')}")
            print(f"    final_box  : {rec.get('final_box')}")
            for h in (rec.get("reply_heads") or []):
                print(f"    tail       : {h!r}")
            print()
    print(f"共 {shown} 条")


if __name__ == "__main__":
    main()
