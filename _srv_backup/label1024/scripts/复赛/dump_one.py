# -*- coding: utf-8 -*-
"""按字段名逐个查看诊断记录，长字符串**转义后再打印**（避免原始换行/CR 破坏终端）。

注意：`reply_heads` / `replies` 是模型原始输出，内含 \n、\r 甚至控制字符。
直接 print 会让光标回车覆盖前面的行，造成"字段错位"的假象。故这里一律
用 ascii() 转义。
"""
from __future__ import annotations

import argparse
import json


def esc(v, width=400):
    if isinstance(v, str):
        s = ascii(v)
    else:
        s = ascii(v)
    if len(s) > width:
        return s[:width] + f"...<total {len(s)}>"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--query-id", nargs="+", default=None)
    ap.add_argument("--list-odd", action="store_true",
                    help="只列出非 matched 的 query_id（不打印内容）")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()

    want = set(args.query_id or [])
    n = 0
    odd = []
    with open(args.diag, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            if rec.get("matched") is not True:
                odd.append((rec.get("query_id"),
                            (rec.get("reply_lens") or []),
                            (rec.get("brace_unclosed") or []),
                            len(rec.get("parsed_pairs") or []),
                            len(rec.get("unmatched_labels") or [])))
            if want and rec.get("query_id") in want:
                print("=" * 72)
                for k, v in rec.items():
                    print(f"{k:18s} = {esc(v)}")
    print(f"\n总记录 {n}；非 matched {len(odd)}")
    if args.list_odd:
        print("query_id             reply_lens        brace  pairs  unmatched")
        for qid, lens, brace, np_, nu in odd:
            print(f"{qid:20s} {str(lens):17s} {str(brace):6s} {np_:5d}  {nu:5d}")


if __name__ == "__main__":
    main()
