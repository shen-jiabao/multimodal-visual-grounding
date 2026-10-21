# -*- coding: utf-8 -*-
"""判定 92 条中性占位框的成因：模型"没输出框"(截断) 还是 "输出了但匹配失败"。

做法：只用复赛归档内的 submission.json（自带 visible/query/bbox，5690 条全量）。
对每个含占位框的图，把该图 query 按序号排序，列出每条 query 的 bbox 及其
"同图是否重复"。若占位集中出现在序号尾部、且尾部之前(query 数 - 占位数)条
恰好落在互不相同的框上 => 模型输出在此处被截断（框压根没生成）；
若占位处之前已出现"框数 >= 该位置序号"的重复/富余框 => 是匹配失败而非截断。
"""
import argparse
import json
import tarfile
from collections import Counter, defaultdict

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
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    sub = load_submission(args.archive)

    # 按 visible 分组，组内按 query_id 序号排序
    by_img = defaultdict(list)
    for qid, rec in sub.items():
        by_img[rec["visible"]].append((qid, rec["query"], tuple(rec["bbox"])))

    n_ph = sum(1 for r in sub.values() if is_placeholder(r["bbox"]))
    print(f"总条目 {len(sub)}  占位框 {n_ph}  图片数 {len(by_img)}")

    bad_imgs = []
    tail_only = 0        # 占位全部位于末尾
    k_before_equals = 0  # 占位前框数与序号恰好吻合
    extra_before = 0     # 占位之前已有重复框 => 更像匹配失败
    for vis, items in by_img.items():
        items.sort(key=lambda x: x[0])
        ph_idx = [i for i, it in enumerate(items) if is_placeholder(it[2])]
        if not ph_idx:
            continue
        n = len(items)
        distinct_before = len({it[2] for it in items[: ph_idx[0]]})
        is_tail = ph_idx == list(range(ph_idx[0], n))
        if is_tail:
            tail_only += 1
        if distinct_before == ph_idx[0]:
            k_before_equals += 1
        if distinct_before < ph_idx[0]:
            extra_before += 1
        # 该图全图不同框数
        bad_imgs.append({
            "visible": vis, "n_queries": n, "n_placeholder": len(ph_idx),
            "placeholder_positions": [i + 1 for i in ph_idx],
            "tail_contiguous": is_tail,
            "distinct_boxes_before_first_ph": distinct_before,
            "distinct_boxes_total": len({it[2] for it in items}),
        })

    print(f"含占位图的图数 {len(bad_imgs)}")
    print(f"占位全部位于 query 序列末尾的图数: {tail_only} / {len(bad_imgs)}")
    print(f"占位前不同框数 == 该位置序号(即此前一一对应): {k_before_equals} / {len(bad_imgs)}")
    print(f"占位前已出现重复框(更像匹配失败): {extra_before} / {len(bad_imgs)}")

    print("\n占位图明细 (按占位数降序)：")
    for r in sorted(bad_imgs, key=lambda x: (-x["n_placeholder"], -x["n_queries"]))[:25]:
        name = r["visible"].split("/")[-1]
        print(f"  {name}  q={r['n_queries']:2d} ph={r['n_placeholder']} "
              f"pos={r['placeholder_positions']} tail={r['tail_contiguous']} "
              f"distinct_before={r['distinct_boxes_before_first_ph']} distinct_total={r['distinct_boxes_total']}")

    # 补充：这些图里占位之前的 query 是否是"细粒度/遮挡类"长文本
    print("\n占位 query 文本长度统计：")
    lens = [len(r["query"]) for r in sub.values() if is_placeholder(r["bbox"])]
    all_lens = [len(r["query"]) for r in sub.values()]
    if lens:
        import statistics
        print(f"  占位 query 字符数: mean={statistics.mean(lens):.1f} median={statistics.median(lens):.0f} min={min(lens)} max={max(lens)}")
        print(f"  全部 query 字符数: mean={statistics.mean(all_lens):.1f} median={statistics.median(all_lens):.0f}")

    # 补充：query 字符数 vs prompt 内累积字符数（截断预算代理）
    print("\n每图 query 总字符数分位（同一 prompt 的文本预算代理）：")
    totals = sorted(sum(len(q) for _, q, _ in v) for v in by_img.values())
    import statistics
    for p in (50, 75, 90, 95, 99):
        print(f"  p{p} = {totals[int(len(totals) * p / 100)]}")
    print(f"  max = {totals[-1]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(bad_imgs, f, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.json}")


if __name__ == "__main__":
    main()
