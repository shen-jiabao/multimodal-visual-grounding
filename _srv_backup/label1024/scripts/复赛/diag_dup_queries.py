# -*- coding: utf-8 -*-
"""同图「文本完全相同」query 的行为审计（只读）。

背景：`multi_box_id=true` 下提示词是 `N. <query>` 的编号列表。若同图内有两条
query **文本逐字相同**，模型可能只回一个 `id`（它「看到的是同一个目标」），
此时另一条 query 会落空 → 占位框。

本脚本读已有预测（基线），对每组同文本 query 回答三个问题：

  1. 组内**是否两条都拿到了真框**（还是有一条落成占位框 / 缺失）；
  2. 组内框是否**完全一致**（一致 = 模型把两条当同一目标，若真值也同目标则正确）；
  3. 组内框的 IoU 分布。

再顺带给出「组内框完全一致」的**可修复空间**：若同组内一条有效、一条占位，
把有效框复制过去是**弱占优**（占位框与任何真值 IoU ≈ 0）。

用法::

    python scripts/复赛/diag_dup_queries.py -p outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz
    python scripts/复赛/diag_dup_queries.py -p <run_dir|predictions.json> --json outputs/复赛/diag/dup_queries.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.复赛.provable_errors import (  # noqa: E402
    FALLBACK,
    load_predictions,
)


def iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = ua + ub - inter
    return float(inter / den) if den > 0 else 0.0


def is_placeholder(box, tol: float = 1e-6) -> bool:
    if box is None:
        return True
    return all(abs(float(b) - f) <= tol for b, f in zip(box, FALLBACK))


def main() -> None:
    ap = argparse.ArgumentParser(description="同图同文本 query 行为审计（只读）")
    ap.add_argument("-p", "--predictions", required=True, help="predictions.json / 归档目录 / run_*.tar.gz")
    ap.add_argument("-q", "--queries", default="dataset/queries/queries.json")
    ap.add_argument("--json", default=None)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    with open(args.queries, encoding="utf-8") as fh:
        qraw = json.load(fh)
    preds = load_predictions(args.predictions)

    by_scene = collections.defaultdict(list)
    for qid, it in qraw.items():
        by_scene[qid.split("_")[0]].append((qid, it["query"]))

    groups = []
    for scene in sorted(by_scene):
        cnt = collections.Counter(q for _, q in by_scene[scene])
        for text, c in cnt.items():
            if c >= 2:
                qids = [qid for qid, q in by_scene[scene] if q == text]
                groups.append((scene, text, qids))

    n_q = sum(len(g[2]) for g in groups)
    print("=" * 78)
    print(f"同图同文本组：{len(groups)} 组 / {n_q} 条 query"
          f"（占 {len(qraw)} 条的 {100.0 * n_q / len(qraw):.2f}%）")
    print("=" * 78)

    both_valid = 0
    one_ph = 0
    all_ph = 0
    identical = 0
    ious = []
    fixable = []          # (qid_ph, qid_src, box_src, scene, text)
    details = []

    for scene, text, qids in groups:
        boxes = [preds.get(q) for q in qids]
        ph = [is_placeholder(b) for b in boxes]
        n_ph = sum(ph)
        if n_ph == 0:
            both_valid += 1
        elif n_ph == len(qids):
            all_ph += 1
        else:
            one_ph += 1

        real = [b for b in boxes if not is_placeholder(b)]
        if real and all(iou(real[0], b) > 0.999 for b in real):
            identical += 1
        if len(real) >= 2:
            ious.append(iou(real[0], real[1]))

        # 弱占优修复：占位的那条可以复制同组唯一有效框
        if 0 < n_ph < len(qids) and real:
            for q, b, p in zip(qids, boxes, ph):
                if p:
                    fixable.append({"query_id": q, "source": qids[boxes.index(real[0])],
                                    "box": real[0], "scene": scene, "text": text})

        details.append({
            "scene": scene, "text": text, "qids": qids,
            "placeholders": int(n_ph),
            "iou_pair": None if len(real) < 2 else round(iou(real[0], real[1]), 4),
            "boxes": [None if b is None else [round(float(v), 4) for v in b] for b in boxes],
        })

    print(f"  组内两条都是真框      {both_valid:3d} / {len(groups)}")
    print(f"  组内一条占位一条真框  {one_ph:3d} / {len(groups)}   ← 可弱占优修复")
    print(f"  组内全是占位          {all_ph:3d} / {len(groups)}")
    print(f"  组内真框完全一致      {identical:3d} / {len(groups)}")
    if ious:
        ious.sort()
        print(f"  组内两真框 IoU  中位 {ious[len(ious) // 2]:.3f}  "
              f"min {ious[0]:.3f}  max {ious[-1]:.3f}")
        print(f"      IoU>0.5 的 {sum(1 for v in ious if v > 0.5)} / {len(ious)}")

    print(f"\n★ 弱占优修复空间（占位框 → 复制同组唯一有效框）：{len(fixable)} 条 query")
    for f in fixable[: args.examples]:
        print(f"      {f['query_id']}  ←  {f['source']}   {f['text'][:56]!r}")
        print(f"          box = {[round(float(v), 4) for v in f['box']]}")

    print("\n明细（前 %d 组）：" % args.examples)
    for d in details[: args.examples]:
        print(f"  {d['scene']}  ph={d['placeholders']}  iou={d['iou_pair']}")
        print(f"      {d['qids']}")
        print(f"      {d['text'][:72]!r}")
        for q, b in zip(d["qids"], d["boxes"]):
            print(f"        {q}  {b}")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "groups": len(groups), "queries": n_q,
                "both_valid": both_valid, "one_placeholder": one_ph,
                "all_placeholder": all_ph, "identical": identical,
                "fixable": fixable, "details": details,
            }, fh, ensure_ascii=False, indent=1)
        print(f"\n已导出：{args.json}")


if __name__ == "__main__":
    main()