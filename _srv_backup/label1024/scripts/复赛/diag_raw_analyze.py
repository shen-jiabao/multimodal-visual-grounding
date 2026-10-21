# -*- coding: utf-8 -*-
"""诊断明细归因：区分「512 token 截断」与「label 回显失败」。

输入：`scripts/infer.py` 在 `cfg.inference.diag_dump=true` 时写出的
`outputs/diag/diag_raw_*.jsonl`（每 query 一行，字段见
`src/inference/predictor.py` 的 `run_inference`）。

要回答三个问题（离线即可定论，不用再花提交次数去试）：

  Q1 占位/丢框的成因。两种解释的判别证据：
     - **截断**：`reply_lens ≈ max_new_tokens`，且 reply 末尾是**不完整**的 JSON
       （`{` 多于 `}`、末尾没有闭合括号）；
     - **回显失败**：reply 完整闭合，`parsed_pairs` 里有 `{label,box}`，
       但 `unmatched_labels` 非空 —— 即模型给了框，只是 label 文本对不上 query。
  Q2 解析抢救的收益上限：`unmatched_labels` 非空的 query 数（这部分框现在被
     `qwen3_vl.py` 的 `continue` 静默丢掉）。
  Q3 分块效应：`chunk_size` 越大是否 unmatched / 截断越严重
     （对应 `max_queries_per_prompt=20` 这个配置是否偏大）。

另外校验"落盘未改变预测"：把 JSONL 里的 `final_box` 与当次
`predictions.json` 逐条比对，不一致条数应为 0。

用法：
    python scripts/复赛/diag_raw_analyze.py --diag outputs/diag/diag_raw_predictions_复赛.jsonl
    python scripts/复赛/diag_raw_analyze.py --diag <jsonl> --predictions <predictions.json>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.matcher import normalize_text  # noqa: E402

# 判据常量
TAIL_RATIO = 0.95      # reply 长度达到 max_new_tokens 的 95% 视为"疑似吃满预算"
TRUNC_TOL = 0.20       # 达到上限 95% 即算


def iter_records(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def brace_balance(text: str) -> int:
    """返回 未闭合的 `{` 数量（>0 表示 JSON 很可能被截断）。"""
    return text.count("{") - text.count("}")


def classify_record(rec, max_new_tokens):
    """给单条 query 判定"丢框原因"。

    判据（按证据强度排序）：
      1. `brace_unclosed[i] > 0`（回复里有未闭合的 `{`）→ **截断的直接证据**；
      2. `unmatched_labels` 非空 → 模型给了框但 label 回显对不上（echo_fail）；
      3. `reply_lens` 吃满 max_new_tokens → 疑似截断；
      4. `parsed_pairs` 有内容但 matched=False → 坐标非法/被丢弃。
    """
    lens = rec.get("reply_lens") or []
    pairs = rec.get("parsed_pairs") or []
    unmatched = rec.get("unmatched_labels") or []
    brace = rec.get("brace_unclosed") or []
    got_box = rec.get("matched") is True
    if got_box:
        return "matched"
    if not lens:
        return "no_reply_record"          # 该 query 所在块没有诊断（异常）
    if any(int(b) > 0 for b in brace):
        return "truncated_json"           # 直接证据：JSON 未闭合
    if unmatched:
        return "echo_fail"                # 有 (label,box) 但 label 对不上
    if max(lens) >= max_new_tokens * TAIL_RATIO:
        return "truncated_budget"         # 吃满 token 预算且没解析出任何 pair
    if pairs:
        return "parse_or_invalid"         # 有 pair 但坐标非法/全部被丢
    return "no_pairs"                     # 回复短但一个 {..} 都没有


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True, help="diag_raw_*.jsonl")
    ap.add_argument("--predictions", default=None, help="可选：校验落盘未改变预测")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--fallback-bbox", default="0.49,0.49,0.51,0.51",
                    help="infer.py 写入的占位框；diag 的 None 与之等价")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    recs = list(iter_records(args.diag))
    print(f"读入 {len(recs)} 条 query 诊断记录")

    reasons = Counter()
    by_chunk = defaultdict(Counter)
    echo_fail_examples = []
    trunc_examples = []
    incomp = 0
    for rec in recs:
        cause = classify_record(rec, args.max_new_tokens)
        reasons[cause] += 1
        by_chunk[rec.get("chunk_size", 0)][cause] += 1
        # 回复完整性（用最后一条回复做代表）
        replies = rec.get("replies") if "replies" in rec else None
        if replies is None:
            # predictor 没把 replies 原样带出时，用 reply_lens 兜底
            replies = []
        if cause == "echo_fail" and len(echo_fail_examples) < 8:
            echo_fail_examples.append({
                "query": rec["query"][:90],
                "unmatched_labels": [normalize_text(u)[:90] for u in
                                     (rec.get("unmatched_labels") or [])][:3],
                "reply_lens": rec.get("reply_lens"),
            })
        if cause in ("truncated_json", "truncated_budget") \
                and len(trunc_examples) < 8:
            trunc_examples.append({
                "query": rec["query"][:90],
                "reply_lens": rec.get("reply_lens"),
                "brace_unclosed": rec.get("brace_unclosed"),
                "head_tail": (rec.get("reply_heads") or [""])[-1][-90:],
            })
        if cause in ("truncated_json", "truncated_budget", "no_pairs",
                     "parse_or_invalid"):
            incomp += 1
    total = len(recs) or 1
    print("\n=== 丢框成因分布 ===")
    for k, v in reasons.most_common():
        print(f"  {k:18s} {v:5d}  ({100.0 * v / total:.2f}%)")
    print("\n判读规则：")
    print("  truncated_json   = 回复里有未闭合的 `{`（JSON 被截断的直接证据）")
    print("  truncated_budget = 回复吃满 max_new_tokens 且未解析出任何 {label,bbox}")
    print("  echo_fail        = 解析出了 (label,bbox)，但 label 文本匹配不到任何 query")
    print("  parse_or_invalid = 有 pair 但坐标非法/被丢弃")
    print("  no_pairs         = 回复不长且一个完整 JSON 对象都没有")
    print("  → 若 echo_fail 占比高，说明占位/丢框主因是 label 回显，"
          "解析抢救（位置+语义二次匹配）有明确收益；")
    print("  → 若 truncated_* 占比高，则应调大 max_new_tokens 或减小 "
          "max_queries_per_prompt（分块）。")

    print("\n=== 按分块大小 ===")
    for cs in sorted(by_chunk):
        row = by_chunk[cs]
        print(f"  chunk_size={cs:2d}  总 {sum(row.values()):5d}  "
              + "  ".join(f"{k}={v}" for k, v in row.most_common()))

    if echo_fail_examples:
        print("\n=== echo_fail 示例（模型给了框但 label 对不上）===")
        for e in echo_fail_examples:
            print(f"  Q: {e['query']}")
            print(f"     reply_lens={e['reply_lens']}")
            for u in e["unmatched_labels"]:
                print(f"     label→ {u}")

    if trunc_examples:
        print("\n=== truncated 示例 ===")
        for e in trunc_examples:
            print(f"  Q: {e['query']}  reply_lens={e['reply_lens']}"
                  f"  brace_unclosed={e.get('brace_unclosed')}")

    # 分块大小 vs 失败率
    print("\n=== 分块大小 vs 丢框率 ===")
    for cs in sorted(by_chunk):
        row = by_chunk[cs]
        tot = sum(row.values())
        bad = tot - row.get("matched", 0)
        print(f"  chunk_size={cs:2d}  丢框 {bad:4d}/{tot:5d} = {100.0 * bad / max(1, tot):.2f}%")

    # 校验落盘未改变预测
    # 注意：predictor 里未匹配的 query 记为 None，而 scripts/infer.py 在写
    # predictions.json 时把 None 替换成 cfg.postprocess.fallback_bbox（占位框）。
    # 因此 "diag 为 None 且 predictions 为占位框" 是**预期**的，不算不一致。
    if args.predictions:
        with open(args.predictions, encoding="utf-8") as fh:
            pred = json.load(fh)
        fb_ref = [float(v) for v in args.fallback_bbox.split(",")]

        def _is_fb(b):
            return (isinstance(b, (list, tuple)) and len(b) == 4
                    and max(abs(float(x) - y) for x, y in zip(b, fb_ref)) <= 1e-6)

        diff = 0
        missing = 0
        substituted = 0
        for rec in recs:
            qid = rec["query_id"]
            fb = rec.get("final_box")
            if qid not in pred:
                missing += 1
                continue
            a, b = fb, pred[qid]
            if a is None and _is_fb(b):
                substituted += 1          # 预期：None -> 占位框
                continue
            if a is None or b is None:
                if a != b:
                    diff += 1
                continue
            if max(abs(float(x) - float(y)) for x, y in zip(a, b)) > 1e-6:
                diff += 1
        print(f"\n=== 落盘一致性校验 ===")
        print(f"  diag.final_box 与 predictions.json 不一致：{diff} 条（应为 0）；"
              f"predictions 中缺 {missing} 条")
        print(f"  None -> 占位框替换：{substituted} 条（预期，= 丢框数）")

    if args.json:
        payload = {"total": len(recs), "reasons": dict(reasons),
                   "by_chunk": {str(k): dict(v) for k, v in by_chunk.items()},
                   "incomplete_replies": incomp}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
