# -*- coding: utf-8 -*-
"""detect-then-select 的离线守门器：**纯重排能在本地证明什么、不能证明什么**。

背景
----
复赛 19.4% 的 query 含序数词。P2-1「detect-then-select」的想法是：

    第 1 步  模型只负责「枚举所有 <NP>」——它擅长的感知任务；
    第 2 步  按框中心（或深度 ROI 均值）确定性排序取第 k 个——纯计算。

第 2 步不需要模型，于是很容易产生一种错觉：**只要本地跑一遍第 2 步，
就能知道这个方案值不值**。本脚本存在的目的正是拆穿这个错觉。

为什么本地模拟**不能**验证 P2-1
-------------------------------
第 1 步（枚举）的产物在本地根本不存在。退而求其次，用模型**已有的**
预测当候选集，会让模拟退化成「把模型自己给的框重新排一遍」——
这正是 ``chain_enforce`` 翻车的那件事：

  · 模型的候选框本身已经是**去重**的（同一 NP 的兄弟 query 很少共框），
    所以「修共框冗余」几乎没有可修的东西；
  · 一旦候选集**少于**最大 rank（实例没枚举全），越界 rank 会被夹到端点，
    反而把两条 query 挤到同一个框上 —— 冗余是**增加**的。

于是本脚本的输出不是「P2-1 好不好」，而是三条**可复用的判定**：

  1. 位移比例：确定性排序与模型原判不一致的比例（= 硬币）。
     若它接近 20%，说明模型的序数判断与几何排序差不多各对一半，
     本地无法分辨谁对 —— 必须靠第 1 步引入**新证据**才有信息增量。
  2. 冗余方向：重排是**减少**还是**增加**共框冗余。增加即否决纯重排。
  3. 位移落点：位移目标是「兄弟 query 原来的框」（纯排列）还是新位置。
     前者说明模型只是把序号搞混、实例都找到了 —— 这才是 P2-1 可能有效的
     唯一形态：**实例齐全、只需要重新指派**。

用法
----
    python scripts/复赛/detect_select_sim.py
    python scripts/复赛/detect_select_sim.py -p <predictions.json|dir|run_*.tar.gz>
    python scripts/复赛/detect_select_sim.py --json out.json --top 20

本脚本只读，不修改任何结果文件；不加载模型，不联网。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference.detect_select import box_centers, extract_target, select_index  # noqa: E402

# 默认对照：09-24 基线（官方 ACC@0.5 = 0.7292）
DEFAULT_RUN = "run_20260924_090232_qwen_multiimg"


def load_predictions(path: str) -> dict:
    """读取预测结果：直接 json / 归档目录 / run_*.tar.gz 三种形态。"""
    if path.endswith((".tar.gz", ".tgz")):
        import tarfile
        with tarfile.open(path, "r:gz") as tf:
            name = next((m.name for m in tf.getmembers()
                         if m.name.endswith("predictions.json")), None)
            if name is None:
                raise SystemExit(f"[error] 压缩包内找不到 predictions.json: {path}")
            with tf.extractfile(name) as fh:
                return json.loads(fh.read().decode("utf-8"))
    if os.path.isdir(path):
        cand = os.path.join(path, "predictions.json")
        if os.path.exists(cand):
            with open(cand, encoding="utf-8") as fh:
                return json.load(fh)
        tgz = sorted(f for f in os.listdir(path) if f.endswith(".tar.gz"))
        if tgz:
            return load_predictions(os.path.join(path, tgz[0]))
        raise SystemExit(f"[error] 目录内找不到 predictions.json: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sec(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def build_groups(pred: dict, qraw: dict, kinds: tuple = ("pos",)) -> list:
    """把同图、同 NP、同轴同向、**rank 互不相同**的 query 归为一个「序数兄弟组」。"""
    by_img = defaultdict(list)
    for qid, v in qraw.items():
        t = extract_target(v["query"])
        if t is None or t["constraint"] is None:
            continue
        c = t["constraint"]
        if c["kind"] not in kinds or c["rank"] is None:
            continue
        box = pred.get(qid)
        if not box:
            continue
        by_img[v["visible"]].append(
            (qid, t["np"].lower(), c, tuple(round(float(x), 6) for x in box)))

    groups = []
    for vis, items in by_img.items():
        buckets = defaultdict(list)
        for qid, np_, c, box in items:
            buckets[(np_, c["axis"], c["direction"])].append((qid, c["rank"], box))
        for key, mem in buckets.items():
            if len(mem) < 2 or len({r for _, r, _ in mem}) < 2:
                continue
            groups.append((vis, key, mem))
    return groups


def simulate(pred: dict, qraw: dict, kinds: tuple = ("pos",)) -> dict:
    """对每个兄弟组做一次确定性重排，统计位移与冗余变化。"""
    groups = build_groups(pred, qraw, kinds)
    n = len(qraw)
    stat = Counter()
    detail = []

    for vis, (np_, axis, direction), mem in groups:
        boxes = [b for _, _, b in mem]
        uniq = sorted(set(boxes), key=lambda b: box_centers([b], axis)[0])
        centers = box_centers(uniq, axis)

        stat["shared_before"] += sum(c - 1 for c in Counter(boxes).values() if c >= 2)

        new = {}
        for qid, rank, box in mem:
            idx = select_index(centers, {"axis": axis, "direction": direction,
                                         "rank": rank, "kind": "pos"})
            new[qid] = uniq[idx] if idx is not None else box

        stat["shared_after"] += sum(c - 1 for c in Counter(new.values()).values() if c >= 2)
        stat["members"] += len(mem)
        moved = [q for q, _, b in mem if new[q] != b]
        stat["moved"] += len(moved)
        stat["kept"] += len(mem) - len(moved)
        if moved:
            stat["groups_with_move"] += 1
        for qid, rank, box in mem:
            if new[qid] == box:
                continue
            others = {b for q2, _, b in mem if q2 != qid}
            key = "moved_to_sibling_box" if new[qid] in others else "moved_to_other"
            stat[key] += 1
        if len(uniq) < max(r for _, r, _ in mem):
            stat["groups_short_candidates"] += 1
        if moved and len(detail) < 12:
            detail.append({"visible": vis, "np": np_, "axis": axis,
                           "direction": direction, "members": mem,
                           "unique": uniq, "new": new})

    return {"queries": n, "groups": len(groups), "stat": stat, "detail": detail,
            "group_sizes": Counter(len(m) for _, _, m in groups),
            "np_groups": len({(v, k[0]) for v, k, _ in groups}),
            "images": len({v for v, _, _ in groups})}


def print_report(name: str, r: dict) -> None:
    st, n = r["stat"], r["queries"]
    print(f"\n----- {name} -----")
    print("  序数兄弟组                     %5d 组" % r["groups"])
    print("  组内 query 总数                 %5d 条 (%.2f%%)"
          % (st["members"], 100.0 * st["members"] / n))
    print("  组内共框冗余（改前，必有错）    %5d" % st["shared_before"])
    print("  组内共框冗余（改后）            %5d   -> %+d"
          % (st["shared_after"], st["shared_after"] - st["shared_before"]))
    m = max(st["members"], 1)
    print("  框未变（保号）                  %5d  (%.1f%%)"
          % (st["kept"], 100.0 * st["kept"] / m))
    print("  框位移                          %5d  (%.1f%%)"
          % (st["moved"], 100.0 * st["moved"] / m))
    print("    · 位移到**兄弟原来的框**      %5d   （纯重排，实例已知）"
          % st["moved_to_sibling_box"])
    print("    · 位移到其他位置              %5d   （风险最高）" % st["moved_to_other"])
    print("  候选实例数 < 最大 rank 的组      %5d 组  <- 越界夹取会制造共框"
          % st["groups_short_candidates"])
    verdict = ("否决纯重排：制造冗余多于修复" if st["shared_after"] > st["shared_before"]
               else "纯重排未制造冗余，但仍无信息增量")
    print("  判定：%s" % verdict)


def main() -> None:
    ap = argparse.ArgumentParser(description="detect-then-select 离线模拟（纯重排的守门器）")
    ap.add_argument("--root", default="dataset", help="数据根（含 queries/queries.json）")
    ap.add_argument("-p", "--predictions", default=None,
                    help="预测结果；缺省用 outputs/复赛/archive 的 09-24 基线")
    ap.add_argument("--json", default=None, help="导出统计结果")
    ap.add_argument("--top", type=int, default=0, help="打印前 N 个位移样例")
    ap.add_argument("--kinds", default="pos",
                    help="参与模拟的约束类型，逗号分隔（pos/depth/middle）")
    args = ap.parse_args()

    qpath = os.path.join(os.path.abspath(args.root), "queries", "queries.json")
    with open(qpath, encoding="utf-8") as fh:
        qraw = json.load(fh)

    pred_path = args.predictions
    if pred_path is None:
        pred_path = os.path.join(ROOT, "outputs", "复赛", "archive", DEFAULT_RUN + ".tar.gz")
        if not os.path.exists(pred_path):
            raise SystemExit(f"[error] 找不到默认基线，请用 -p 指定: {pred_path}")
    pred = load_predictions(pred_path)
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())

    print("预测来源: " + pred_path)
    print("约束类型: " + ",".join(kinds))
    r = simulate(pred, qraw, kinds)
    print_report("确定性重排模拟", r)

    sec("位移样例（模型只是把序号搞混，还是实例根本没找到？）")
    shown = args.top or 8
    for d in r["detail"][:shown]:
        print("\n  [%s]  np=%r  axis=%s dir=%+d"
              % (os.path.basename(d["visible"]), d["np"], d["axis"], d["direction"]))
        print("     候选实例（按中心排序）:")
        for i, b in enumerate(d["unique"]):
            print("        #%d %s" % (i + 1, [round(v, 3) for v in b]))
        for qid, rank, box in sorted(d["members"], key=lambda t: t[1]):
            tag = "   " if d["new"][qid] == box else " ->"
            print("       rank=%d %-14s 原=%s%s新=%s"
                  % (rank, qid, [round(v, 3) for v in box], tag,
                     [round(v, 3) for v in d["new"][qid]]))

    sec("组大小分布（决定枚举调用的摊薄率）")
    for k in sorted(r["group_sizes"]):
        print("     %2d 条/组 : %5d 组  (%d 条 query)"
              % (k, r["group_sizes"][k], k * r["group_sizes"][k]))
    print("     去重 NP 组数 = %d（若每 NP 一次枚举调用）" % r["np_groups"])
    print("     涉及的图数   = %d" % r["images"])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"predictions": pred_path, "queries": r["queries"],
                       "groups": r["groups"], "stat": dict(r["stat"]),
                       "group_sizes": {str(k): v for k, v in r["group_sizes"].items()},
                       "np_groups": r["np_groups"], "images": r["images"]},
                      fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
