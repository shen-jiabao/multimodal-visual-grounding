# -*- coding: utf-8 -*-
"""把「本轮 diag 的链修正前框」与「0.7292 基线预测」逐条对比，隔离链修正的真实影响。

要回答三件事：

  Q1 本轮（chain=project）的预测，去掉链修正后，是否与 0.7292 那一版**逐条一致**？
     - 若一致 → 两次提交只差「链修正」一个变量，本轮提交的分数就是链修正的净效应，
       可直接做官方 A/B（不用再猜）。
     - 若不一致 → 说明还有别的变化（确定性/版本/渲染），需要先解释清楚再比。

  Q2 链修正本身在提交层面动了多少条、动了多少（p50/p90/max）。

  Q3 两次提交的占位框数是否相同（链修正不应改变占位数）。

用法：
  python scripts/复赛/compare_runs.py \
      --baseline outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \
      --new      outputs/复赛/archive/run_20260928_003617_qwen_diag/predictions.json \
      --diag     outputs/复赛/archive/diag_raw_predictions_qwen_diag.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
from collections import Counter

PLACEHOLDER = (0.49, 0.49, 0.51, 0.51)


def load_any(path):
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            cands = [m for m in t.getmembers()
                     if m.name.endswith("predictions.json") and m.isfile()]
            if not cands:
                cands = [m for m in t.getmembers()
                         if m.name.endswith("submission.json") and m.isfile()]
            with t.extractfile(cands[0]) as fh:
                return json.load(fh)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def box_of(v):
    if isinstance(v, dict):
        return v.get("bbox") or v.get("bbox_2d")
    return v


def is_ph(b):
    return (isinstance(b, (list, tuple)) and len(b) == 4
            and max(abs(float(x) - y) for x, y in zip(b, PLACEHOLDER)) <= 1e-6)


def cmp_boxes(a, b, tol=1e-6):
    if a is None or b is None:
        return a is None and b is None
    return max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def center(b):
    return ((float(b[0]) + float(b[2])) / 2, (float(b[1]) + float(b[3])) / 2)


def report(name_a, a, name_b, b):
    keys = sorted(set(a) | set(b))
    only_a = [k for k in keys if k not in b]
    only_b = [k for k in keys if k not in a]
    # None 与占位框的互转是"同义"的（都表示没有真实检测），不算真实差异
    def _none_ph_equiv(k):
        if k not in a or k not in b:
            return False
        ba, bb = box_of(a[k]), box_of(b[k])
        return ((ba is None and is_ph(bb)) or (is_ph(ba) and bb is None))

    raw_diff = [k for k in keys if k in a and k in b
                and not cmp_boxes(box_of(a[k]), box_of(b[k]))]
    diff = [k for k in raw_diff if not _none_ph_equiv(k)]
    print(f"\n--- {name_a}  vs  {name_b} ---")
    print(f"  条数 {len(a)} vs {len(b)}；只在 A {len(only_a)}、只在 B {len(only_b)}")
    print(f"  框不同（原始）= {len(raw_diff)}；剔除「None↔占位框」后 = {len(diff)}")
    ph_a = sum(1 for k in a if is_ph(box_of(a[k])))
    ph_b = sum(1 for k in b if is_ph(box_of(b[k])))
    none_a = sum(1 for k in a if box_of(a[k]) is None)
    none_b = sum(1 for k in b if box_of(b[k]) is None)
    print(f"  占位框 {ph_a} vs {ph_b}；None {none_a} vs {none_b}")
    none_ph = sum(1 for k in raw_diff if _none_ph_equiv(k))
    if none_ph:
        print(f"  其中 None <-> 占位框 = {none_ph} 条（等价：都没有真实检测）")
    if diff:
        ds = []
        for k in diff:
            ba, bb = box_of(a[k]), box_of(b[k])
            if ba is None or bb is None:
                continue          # None 不能算位移
            ca, cb = center(ba), center(bb)
            ds.append((((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2) ** 0.5, k))
        ds.sort(reverse=True)
        vals = sorted(d for d, _ in ds)
        n = len(vals)
        print(f"  位移 p50={vals[n // 2]:.4f}  p90={vals[min(n - 1, int(n * .9))]:.4f}  max={vals[-1]:.4f}")
        print(f"  位移 >0.15 的 = {sum(1 for v in vals if v > 0.15)}")
        print(f"  最大 5 条：" + ", ".join(f"{k}({d:.3f})" for d, k in ds[:5]))
    return set(diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    base = load_any(args.baseline)
    new = load_any(args.new)
    base = {k: box_of(v) for k, v in base.items()}
    new = {k: box_of(v) for k, v in new.items()}

    pre, fin = {}, {}
    with open(args.diag, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                pre[r["query_id"]] = r.get("pre_chain_box")
                fin[r["query_id"]] = r.get("final_box")

    d1 = report("0.7292 基线", base, "本轮 diag.pre_chain_box（= 关掉链修正）", pre)
    d2 = report("0.7292 基线", base, "本轮 final_box（= 开链修正）", fin)
    d3 = report("本轮 pre_chain_box", pre, "本轮 final_box", fin)
    d4 = report("本轮 final_box(diag)", fin, "本轮 predictions.json", new)

    print("\n=== 结论 ===")
    if not d1:
        print("  Q1: pre_chain_box 与 0.7292 基线**逐条一致** → 两次提交只差「链修正」一个变量。")
        print("      本轮提交的官方分数 = 0.7292 + 链修正净效应，可直接读 A/B。")
    else:
        print(f"  Q1: pre_chain_box 与 0.7292 基线有 {len(d1)} 条不同 → 除链修正外还有别的差异，"
              f"不能直接做 A/B，需先解释这 {len(d1)} 条。")
        print(f"      样例：{sorted(d1)[:5]}")
    print(f"  Q3: 链修正移动 {len(d3)} 条；占位框数"
          f"{sum(1 for v in base.values() if is_ph(v))}(基线) vs "
          f"{sum(1 for v in fin.values() if is_ph(v))}(本轮)")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "baseline_vs_pre_diff": sorted(d1),
                "baseline_vs_final_diff": sorted(d2),
                "pre_vs_final_diff": sorted(d3),
                "diag_final_vs_predictions_diff": sorted(d4),
            }, fh, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
