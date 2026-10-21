# -*- coding: utf-8 -*-
"""用复赛数据本身估计"512 token 输出预算能装下多少条 query"，
并把 92 条占位框的截断点与预算对齐，验证 max_new_tokens 假说。

不依赖模型权重的做法：
  1. 统计占位图里 label 的"标签文本长度"分布；
  2. 统计 label 中出现的 query 自由文本平均长度（复赛 query 自身长度即标签长度代理）；
  3. 用经验比例（英文 ~4 字符/token，加标点/数字更密）估算每条约多少 token；
  4. 求 N 条 query 的总 token 预算，判断 512 是不是截断点。
"""
import argparse
import json
import tarfile
from collections import defaultdict

PLACEHOLDER = (0.49, 0.49, 0.51, 0.51)


def is_placeholder(b, tol=1e-6):
    return all(abs(b[i] - PLACEHOLDER[i]) <= tol for i in range(4))


def load_submission(archive):
    with tarfile.open(archive) as t:
        m = [x for x in t.getmembers() if x.name.endswith("/submission.json")][0]
        return json.load(t.extractfile(m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz")
    ap.add_argument("--max-per-prompt", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    sub = load_submission(args.archive)

    by_img = defaultdict(list)
    for qid, rec in sub.items():
        by_img[rec["visible"]].append((qid, rec["query"], tuple(rec["bbox"])))

    # 只统计被引用场景里每个 prompt 分块(<=max_per_prompt)的累积字符数，
    # 对齐真实调用：predictor 按 split_by_count(texts, 20) 分块。
    chunks = []          # (visible, chunk_index, n_q, chars, avg_chars, ph_count)
    for vis, items in by_img.items():
        items.sort(key=lambda x: x[0])
        for ci in range(0, len(items), args.max_per_prompt):
            part = items[ci:ci + args.max_per_prompt]
            chars = sum(len(q) for _, q, _ in part)
            ph = sum(1 for _, _, b in part if is_placeholder(b))
            chunks.append((vis, ci, len(part), chars, chars / len(part), ph))

    n_chunks = len(chunks)
    n_full = sum(1 for c in chunks if c[2] == args.max_per_prompt)
    print(f"prompt 分块数 {n_chunks}  (满 {args.max_per_prompt} 条的块: {n_full})")

    ph_chunks = [c for c in chunks if c[5] > 0]
    print(f"含占位的块 {len(ph_chunks)}/{n_chunks}")
    if ph_chunks:
        cs = sorted(c[3] for c in ph_chunks)
        print(f"含占位块的 query 总字符数: median={cs[len(cs)//2]} min={cs[0]} max={cs[-1]}")

    # 分成两半对比：含占位块 vs 无占位块，看字符数是否是判据
    ok_chunks = [c for c in chunks if c[5] == 0]
    if ok_chunks:
        cs = sorted(c[3] for c in ok_chunks)
        print(f"无占位块的 query 总字符数: median={cs[len(cs)//2]} p90={cs[int(len(cs)*0.9)]} max={cs[-1]}")

    # token 估算：英文文本经验 ~4.0 字符/token（含数字/标点更差，取 3.6 做保守）
    for cpt in (4.0, 3.6, 3.2):
        print(f"\n--- 假设 {cpt} 字符/token ---")
        per_box_tokens = []  # 每条 query 对应的 JSON 片段 token 估算
        for vis, ci, n, chars, avg, ph in chunks:
            # 每框输出形如 {"label":"<query>","bbox_2d":[123,456,789,1000]} 约 40 固定字符
            total_tok = (chars + 40 * n) / cpt
            if total_tok > args.max_new_tokens:
                pass
        over = sum(1 for vis, ci, n, chars, avg, ph in chunks
                   if (chars + 40 * n) / cpt > args.max_new_tokens)
        print(f"  估算输出 token 超 {args.max_new_tokens} 的块数: {over}/{n_chunks}")
        # 边界：预估"能完整输出几条"
        caps = []
        for vis, ci, n, chars, avg, ph in chunks:
            budget = args.max_new_tokens
            used = 0.0
            k = 0
            for _, q, _ in by_img[vis][ci:ci + n]:
                t = (len(q) + 40) / cpt
                if used + t > budget:
                    break
                used += t
                k += 1
            caps.append((n, k))
        # 全块能放下的比例
        full_ok = sum(1 for n, k in caps if k >= n)
        print(f"  估算能完整输出的块数: {full_ok}/{n_chunks}")

    # 1.436 等异常块定位
    print("\n含占位块明细：")
    for vis, ci, n, chars, avg, ph in sorted(ph_chunks, key=lambda x: -x[5])[:20]:
        print(f"  {vis.split('/')[-1]:>12}  chunk{ci//args.max_per_prompt} n={n:2d} "
              f"chars={chars:4d} avg={avg:5.1f} ph={ph} tok@3.6={(chars+40*n)/3.6:7.1f}")


if __name__ == "__main__":
    main()
