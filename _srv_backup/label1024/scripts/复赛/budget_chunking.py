# -*- coding: utf-8 -*-
"""离线：把「每图 prompt 分块的字符预算」与「占位/丢框」对齐，定出安全分块阈值。

背景（本脚本之前的离线结论）：
  - 1295 张图里 49 张含占位框（92 条），其中 46/49 图的占位**全部落在该图 query 序列末尾**，
    且 41/49 图满足「占位前不同框数 == 占位起始位置-1」（即模型输出前 k 条就停了）；
  - 占位 query 明显更长：字符数 median 90 vs 全局 median 49；
  - 用 3.6 字符/token 估算，输出预算超 512 的分块只有 4 个，说明「全量截断」不成立，
    但「长 query 分块更容易在末尾掉框」成立。

本脚本回答一个可执行的离线问题：
  若把分块判据从「固定 20 条」换成「累计字符预算 B」，需要多少额外前向调用？
  以及被拆开的分块里，历史上出过多少占位/共框？

用法：
  python scripts/复赛/budget_chunking.py \
      --archive outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \
      --predictions outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \
      --thresholds 200 300 400 500 600
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PLACEHOLDER = (0.49, 0.49, 0.51, 0.51)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUERIES = os.path.join(PROJECT_ROOT, "dataset", "queries", "queries.json")


def is_ph(b, tol=1e-6):
    return b is not None and all(abs(float(b[i]) - PLACEHOLDER[i]) <= tol for i in range(4))


def load_json_from_any(path):
    """支持直接 json / 归档目录 / tar.gz。"""
    if os.path.isfile(path) and tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            cands = [m for m in t.getmembers()
                     if m.name.endswith("predictions.json") and m.isfile()]
            if not cands:
                cands = [m for m in t.getmembers()
                         if m.name.endswith("submission.json") and m.isfile()]
            with t.extractfile(cands[0]) as fh:
                return json.load(fh)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise FileNotFoundError(path)


def load_queries():
    with open(QUERIES, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict) and "queries" in raw:
        raw = raw["queries"]
    rows = []
    if isinstance(raw, list):
        for rec in raw:
            rows.append((rec["query_id"], rec.get("visible") or rec.get("image"),
                         rec.get("query", "")))
    else:
        for qid, rec in raw.items():
            if isinstance(rec, str):
                rows.append((qid, None, rec))
            else:
                rows.append((qid, rec.get("visible") or rec.get("image"),
                             rec.get("query", "")))
    return rows


def chunk_by_count(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)]


def chunk_by_budget(items, budget):
    """贪心（按 query 文本长度累计）：超过 budget 就换块（单条超预算也独占一块）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True, help="归档 tar.gz（取 submission/predictions）")
    ap.add_argument("--max-queries-per-prompt", type=int, default=20)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[200, 250, 300, 350, 400, 500, 600, 800])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    pred = load_json_from_any(args.archive)
    rows = load_queries()
    by_img = defaultdict(list)
    vis_of = {}
    for qid, vis, q in rows:
        vis_of[qid] = vis or "?"
        by_img[vis or "?"].append((qid, q))

    # 归档 predictions 若为 {qid: [x1,y1,x2,y2]}
    def box_of(qid):
        v = pred.get(qid)
        if isinstance(v, dict):
            v = v.get("bbox") or v.get("bbox_2d")
        return v

    # 每块的统计
    def chunk_stats(chunks_of_img):
        """返回 [(n, chars, n_ph, n_shared_queries)]"""
        out = []
        for ch in chunks_of_img:
            chars = sum(len(t) for _, t in ch)
            ph = sum(1 for qid, _ in ch if is_ph(box_of(qid)))
            # 该块内 query 共框（同块内多 query 拿到完全相同的框）
            cnt = defaultdict(int)
            for qid, _ in ch:
                b = box_of(qid)
                if b is not None and not is_ph(b):
                    cnt[tuple(round(float(x), 6) for x in b)] += 1
            shared = sum(c - 1 for c in cnt.values() if c > 1)
            out.append((len(ch), chars, ph, shared))
        return out

    base_chunks = []
    for vis, items in by_img.items():
        items.sort(key=lambda x: x[0])
        base_chunks += chunk_stats(chunk_by_count(items, args.max_queries_per_prompt))

    n_calls = len(base_chunks)
    tot_ph = sum(c[2] for c in base_chunks)
    tot_shared = sum(c[3] for c in base_chunks)
    print("=== 当前配置（固定 %d 条/块）===" % args.max_queries_per_prompt)
    print(f"模型调用次数（每图分块数合计）= {n_calls}")
    print(f"占位框 = {tot_ph}   块内共框 query = {tot_shared}")

    # 字符数分桶 × 占位率
    buckets = [(0, 200), (200, 300), (300, 400), (400, 500), (500, 700), (700, 10 ** 9)]
    print("\n=== 按「块内 query 总字符数」分桶 ===")
    print(f"{'字符区间':>14} {'块数':>5} {'query数':>7} {'占位':>5} {'占位率':>7} {'块内共框':>8}")
    for lo, hi in buckets:
        sel = [c for c in base_chunks if lo <= c[1] < hi]
        if not sel:
            continue
        nq = sum(c[0] for c in sel)
        ph = sum(c[2] for c in sel)
        sh = sum(c[3] for c in sel)
        label = "%d-%s" % (lo, "+" if hi >= 10 ** 9 else str(hi))
        print(f"{label:>14} {len(sel):>5} {nq:>7} "
              f"{ph:>5} {ph / nq * 100:>6.2f}% {sh:>8}")

    print("\n=== 换成「累计字符预算」需要多少调用 ===")
    print(f"{'预算':>6} {'调用次数':>7} {'相对增幅':>8} {'涉及历史占位':>12} {'涉及历史共框':>12}")
    result = {"base": {"calls": n_calls, "placeholder": tot_ph, "shared": tot_shared},
              "grid": []}
    for thr in args.thresholds:
        chunks = []
        for vis, items in by_img.items():
            items.sort(key=lambda x: x[0])
            chunks += chunk_stats(chunk_by_budget(items, thr))
        n = len(chunks)
        # 被拆开的分块才计入"涉及"
        print(f"{thr:>6.0f} {n:>7} {(n / n_calls - 1) * 100:>7.1f}% "
              f"{sum(c[2] for c in chunks):>12} {sum(c[3] for c in chunks):>12}")
        result["grid"].append({"budget": thr, "calls": n,
                               "placeholder": sum(c[2] for c in chunks),
                               "shared": sum(c[3] for c in chunks)})

    # 单块最大字符数与超预算块明细
    print("\n=== 当前配置下字符数最大的 15 个块 ===")
    for c in sorted(base_chunks, key=lambda x: -x[1])[:15]:
        print(f"  n={c[0]:2d} chars={c[1]:4d} 占位={c[2]} 共框={c[3]}"
              f"  估算token@3.6={(c[1] + 40 * c[0]) / 3.6:7.1f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
