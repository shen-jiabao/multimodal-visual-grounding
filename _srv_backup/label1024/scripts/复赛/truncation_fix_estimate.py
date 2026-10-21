# -*- coding: utf-8 -*-
"""用本轮 diag 落盘精确回答「截断该怎么修」。

背景：`diag_raw_analyze.py` 已定论——92 条丢框 = 81 truncated_json + 8
    truncated_budget + 2 parse_or_invalid + 1 no_pairs，**echo_fail = 0**。
    即全部是「输出没写完」，不是「label 回显失败」。

本脚本用**真实观测到的 reply 字符长度**反推两件事：

  A. chars/token 实测值。截断块的 reply 恰好被 max_new_tokens 截断，故
     `reply_len / max_new_tokens` 就是该块的 chars/token。用它替代之前
     猜的 3.6/4.0（数字密集型 JSON 远低于此）。

  B. 每个块的「写完需要多少 token」= (query 字符数 + 40*n) / chars_per_token。
     由此得出：
       - 把 max_new_tokens 提到多少可覆盖全部块（零额外前向）；
       - 改成按条数/按字符预算分块后，还剩多少块超预算、要多付多少调用。

用法：
  python scripts/复赛/truncation_fix_estimate.py \
      --diag outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl \
      --json outputs/复赛/diag/truncation_fix.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict

# 每个 {"label":"...","bbox_2d":[a,b,c,d]} 的固定开销字符数
# （键名 + 引号 + 括号 + 逗号 + 4 个坐标，坐标多为 3-4 位）
FIXED_PER_BOX = 40


def iter_records(path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--json", default=None)
    ap.add_argument("--counts", type=int, nargs="+", default=[6, 8, 10, 12, 20])
    ap.add_argument("--budgets", type=int, nargs="+", default=[400, 600, 800, 1000, 1200])
    args = ap.parse_args()

    recs = list(iter_records(args.diag))
    # label 回显失败的总量。独立统计，避免被 classify_record 的优先级
    # （brace_unclosed 先判）掩盖：截断块里也可能同时有 unmatched label。
    n_unmatched_recs = sum(1 for r in recs if r.get("unmatched_labels"))
    n_unmatched_labels = sum(len(r.get("unmatched_labels") or []) for r in recs)
    print(f"读入 {len(recs)} 条 query 诊断记录")
    print(f"含 unmatched_labels 的 query = {n_unmatched_recs} 条；"
          f"label 总数 = {n_unmatched_labels} 个")
    # 按 (visible, chunk_index) 归块，保持块内 query 顺序
    chunks = defaultdict(list)
    for rec in recs:
        chunks[(rec["visible"], rec["chunk_index"])].append(rec)

    rows = []
    ratios = []
    for (vis, ci), rs in chunks.items():
        rs.sort(key=lambda r: r["query_id"])
        n = len(rs)
        chars = sum(len(r["query"]) for r in rs)
        matched = sum(1 for r in rs if r.get("matched"))
        reply_len = max((max(r.get("reply_lens") or [0]) for r in rs), default=0)
        brace = max((max(r.get("brace_unclosed") or [0]) for r in rs), default=0)
        rows.append({
            "visible": vis, "chunk_index": ci, "n": n, "chars": chars,
            "matched": matched, "reply_len": reply_len, "brace_unclosed": brace,
            "truncated": brace > 0,
        })
        # 标定：未闭合的块，reply 恰好在 max_new_tokens 处被切断
        if brace > 0 and reply_len > 0:
            ratios.append(reply_len / args.max_new_tokens)

    if ratios:
        cpt = statistics.median(ratios)
        cpt_lo = sorted(ratios)[max(0, len(ratios) // 4 - 1)]
        print(f"=== chars/token 实测标定（来自 {len(ratios)} 个截断块）===")
        print(f"  median = {cpt:.3f}   p25 = {cpt_lo:.3f}   min = {min(ratios):.3f}"
              f"   max = {max(ratios):.3f}")
        print(f"  （此前报告用 3.6 估算；实测密度更高，故截断比估算的更严重）")
    else:
        cpt = cpt_lo = 3.0
        print(f"无截断块可标定，退用 cpt={cpt}")

    # 逐块自校准：截断块的 reply_len 正是 512 token 的产出，故其 chars/token 是实测值；
    # 未截断块无法标定，用截断块 cpt 的 p25（最保守 → 高估 need_tokens）。
    for r in rows:
        if r["truncated"] and r["reply_len"] > 0:
            r["cpt"] = r["reply_len"] / args.max_new_tokens
            r["cpt_src"] = "observed"
        else:
            r["cpt"] = cpt_lo
            r["cpt_src"] = "conservative"
        need_chars = r["chars"] + FIXED_PER_BOX * r["n"]
        r["need_tokens"] = need_chars / r["cpt"]

    obs_trunc = sum(1 for r in rows if r["truncated"])
    est_trunc = sum(1 for r in rows if r["need_tokens"] > args.max_new_tokens)
    print(f"\n=== 模型校准（当前分块 20 条/块）===")
    print(f"  实测截断块 = {obs_trunc}")
    print(f"  按 need_tokens > {args.max_new_tokens} 预测的截断块 = {est_trunc}")
    print(f"  → {'校准良好' if abs(est_trunc - obs_trunc) <= max(3, obs_trunc * 0.2) else '偏差较大，估算需谨慎'}")

    # ---- A. max_new_tokens 需要提到多少 ----
    needs = sorted(r["need_tokens"] for r in rows)
    print(f"\n=== A. 若只调 max_new_tokens（零额外前向）===")
    for cand in (640, 768, 1024, 1280, 1536):
        over = sum(1 for v in needs if v > cand)
        print(f"  max_new_tokens={cand:5d}  仍超预算的块 = {over:3d}/{len(needs)}")
    print(f"  最大 need_tokens = {needs[-1]:.1f}")
    need_all = needs[-1]
    cover = [c for c in (640, 768, 1024, 1280, 1536) if need_all <= c]
    print(f"  → 覆盖全部块的最小候选 = {cover[0] if cover else '>1536'}")
    print(f"  → 仅被截断的那些块会多解码；未截断块仍按 EOS 提前停止，"
          f"故墙钟时间增量 ≈ {obs_trunc} 块 × 少量 token，可忽略。")

    # ---- A2. 完全经验估计：need_tokens ≈ max_new_tokens × n / k ----
    # 截断块恰好在 max_new_tokens 处被切断，其 reply 里含 k 个完整框（=该块内 matched 的
    # query 数），故"每框 token 数" ≈ max_new_tokens / k，整块写完 ≈ max_new_tokens × n / k。
    # 字符数在此完全约掉，不含任何「40 字符/框」假设，比上面的估算可靠。
    emp = []
    zero_k = []
    for r in rows:
        n, k = r["n"], r["matched"]
        if k <= 0:
            if r["truncated"]:
                zero_k.append(r)
            continue
        emp.append({"visible": r["visible"], "n": n, "k": k,
                    "truncated": r["truncated"],
                    "need": args.max_new_tokens * n / k})
    trunc_emp = [e for e in emp if e["truncated"]]
    if trunc_emp:
        vals = sorted(e["need"] for e in trunc_emp)
        print(f"\n=== A2. 经验估计 need_tokens ≈ {args.max_new_tokens}×n/k（无字符假设）===")
        print(f"  截断块 {len(trunc_emp)} 个；中位 {vals[len(vals) // 2]:.0f} / "
              f"p90 {vals[min(len(vals) - 1, int(len(vals) * 0.9))]:.0f} / max {vals[-1]:.0f}")
        print(f"  {'预算':>6} {'仍超预算的截断块':>16} {'未覆盖的 query':>14}")
        for cand in (640, 768, 1024, 1536):
            over = [e for e in trunc_emp if e["need"] > cand]
            missed = sum(e["n"] - e["k"] for e in over)
            print(f"  {cand:>6} {len(over):>16} {missed:>14}")
        print(f"  （未覆盖的 query 是这些块里预计仍会丢掉的条数上界）")
        worst = sorted(trunc_emp, key=lambda e: -e["need"])[:5]
        print(f"  最吃 token 的截断块：")
        for e in worst:
            print(f"    n={e['n']:2d} 已出框k={e['k']:2d}  需≈{e['need']:.0f} token"
                  f"  {e['visible'].split('/')[-1]}")
    if zero_k:
        print(f"  另有 {len(zero_k)} 个截断块**一个框都没出**（k=0，无法用此法估计），"
              f"其 n = {[r['n'] for r in zero_k]}")

    # ---- B. 换分块（按条数上限）----
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["visible"]].append(r)
    # 用 query 文本重建每图的完整 query 列表
    img_queries = defaultdict(list)
    for rec in recs:
        img_queries[rec["visible"]].append((rec["query_id"], rec["query"]))
    for v in img_queries:
        img_queries[v].sort(key=lambda x: x[0])

    def simulate(pred_chunks):
        """pred_chunks(queries) -> [(n, chars)]；返回 (调用数, 预计超预算块数)"""
        calls = 0
        over = 0
        for v, items in img_queries.items():
            for ch in pred_chunks(items):
                calls += 1
                chars = sum(len(t) for _, t in ch)
                if (chars + FIXED_PER_BOX * len(ch)) / cpt > args.max_new_tokens:
                    over += 1
        return calls, over

    def by_count(items, cap):
        return [items[i:i + cap] for i in range(0, len(items), cap)]

    def by_budget(items, budget):
        out, cur, used = [], [], 0
        for it in items:
            t = it[1]
            if cur and used + len(t) > budget:
                out.append(cur)
                cur, used = [], 0
            cur.append(it)
            used += len(t)
        if cur:
            out.append(cur)
        return out

    base_calls, base_over = simulate(lambda items: by_count(items, 20))
    print(f"\n=== B1. 按「条数上限」分块 ===")
    print(f"  {'上限':>5} {'调用数':>7} {'增幅':>8} {'预计超预算块':>12}")
    print(f"  {20:>5} {base_calls:>7} {0.0:>7.1f}% {base_over:>12}")
    res_counts = []
    for cap in args.counts:
        if cap >= 20:
            continue
        calls, over = simulate(lambda items, c=cap: by_count(items, c))
        print(f"  {cap:>5} {calls:>7} {(calls / base_calls - 1) * 100:>7.1f}% {over:>12}")
        res_counts.append({"cap": cap, "calls": calls, "over": over})

    print(f"\n=== B2. 按「累计字符预算」分块 ===")
    print(f"  {'预算':>5} {'调用数':>7} {'增幅':>8} {'预计超预算块':>12}")
    res_budgets = []
    for b in args.budgets:
        calls, over = simulate(lambda items, bb=b: by_budget(items, bb))
        print(f"  {b:>5} {calls:>7} {(calls / base_calls - 1) * 100:>7.1f}% {over:>12}")
        res_budgets.append({"budget": b, "calls": calls, "over": over})

    # ---- C. 组合：只提 max_new_tokens 不动分块 ----
    print(f"\n=== C. 结论 ===")
    print(f"  1) echo_fail = 0 → 解析抢救无收益，不做。")
    print(f"  2) 全部丢框都是截断 → 提 max_new_tokens 是唯一「零额外前向」的修法；"
          f"估算需 {need_all:.0f} token 才覆盖全部块。")
    print(f"  3) 分块收紧是第二道保险，但每次都要多付前向（见 B1/B2 表）。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "chars_per_token": cpt,
                "max_new_tokens_current": args.max_new_tokens,
                "observed_truncated_chunks": obs_trunc,
                "estimated_truncated_chunks": est_trunc,
                "max_need_tokens": need_all,
                "count_schemes": res_counts,
                "budget_schemes": res_budgets,
            }, fh, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
