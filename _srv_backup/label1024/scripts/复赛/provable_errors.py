# -*- coding: utf-8 -*-
"""可证明错误面板（复赛 · 无 GT 环境下的改动守门器）。

背景
----
复赛 ``queries.json`` 不含 bbox，官方评测只在提交后返回，本地无法直接测
ACC@0.5。早期尝试用「序数一致性」当代理指标（chain_enforce=project）：
纸面破序 42 -> 0 全绿，官方分数反降 0.7292 -> 0.7283 —— 代理指标与目标
指标脱钩，盲改失败。

本脚本提供另一类代理：**可证明错误**。它不做 ACC 预测，只做两件事：

  1) 度量硬损失下界：数出「无论如何都不可能 IoU>=0.5」的条数，给出 ACC 硬下界。
  2) 充当改动守门器：任何改动上线前先跑本脚本。若它移动了大量「有效框」
     而可证明错误净减少很少，则该改动没有可验证的收益依据 —— 等同盲改。

错误分级（已用初赛 9555 条 GT 校准）
------------------------------------
计入面板 —— 严格可证（数学上必错，或组内必有 >=1 条错）：

  placeholder  解析失败 -> 中性占位框 [0.49,0.49,0.51,0.51]：模型未产出预测
  zero_area    宽或高 <= 1e-6，交集面积必为 0，IoU 恒为 0
  conflict     同框的两条 query 约束**强互斥**（同类 + 同 kind + 同轴 + 同向 +
               不同 rank），不可能共指 -> 组内 >=1 条必错

不计入面板 —— 分布异常（曾被当作错误信号，经 GT 校准后**证伪**）：

  thin         0 < 宽或高 < 0.01
  huge         面积 > 0.5
  shared       同类 query 共用同一框

  校准依据（初赛 9555 条 GT，纯框分布统计，不需要图像）：

  | 类别   | 预测（复赛 09-24） | GT（初赛 9555） | 倍数 |
  |--------|------------------:|---------------:|-----:|
  | thin   |  28 / 5598 = 0.50%| 93/9555 = 0.97%| 0.5x |
  | huge   |  42 / 5598 = 0.75%| 44/9555 = 0.46%| 1.6x |
  | shared | 363 / 5690 = 6.38%|449/9555 = 4.70%| 1.4x |

  · thin 在预测里比 GT **更少**（0.5x）—— 把细条当错误，会把「模型比真值更
    保守」误判成错。GT 自身 p10 面积 = 0.00115，细长目标真实存在。
  · shared 在 GT 里本来就有 4.70%：同一目标常被多条 query 指代（如
    "the peacock on the right" 与 "the peacock standing on the far right"
    指向同一只孔雀）。预测仅高出 1.4 倍，属正常现象，不是错误。
  · huge 倍数最高（1.6x）但绝对量仅 42 条，且 GT 里确有 44 条面积 > 0.5 的
    整片结构（栅栏 / 栏杆 / 排队通道），不足以判定必错。

  故三者一律不计入面板，仅作为「分布偏离 GT」的观察项输出。

用法
----
    python scripts/复赛/provable_errors.py -p A.json
    python scripts/复赛/provable_errors.py -p A.json -c B.json --json out.json

``-p`` / ``-c`` 接受三种形态：直接 predictions.json、归档目录、run_*.tar.gz。
本脚本只读，不修改任何结果文件。
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

from src.inference.matcher import parse_spatial_constraint  # noqa: E402
from src.utils.text import clean_query  # noqa: E402

# configs/config.yaml 的 postprocess.fallback_bbox
FALLBACK = (0.49, 0.49, 0.51, 0.51)
TOL = 1e-6
THIN = 0.01
HUGE_AREA = 0.5

# 名词短语抽取时丢弃的修饰/方位/序数词（保留物体类别词）
_MOD = {
    "the", "a", "an", "of", "on", "in", "to", "from", "and", "or", "is", "it",
    "that", "this", "with", "at", "by", "for", "as", "its", "their", "there",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth",
    "leftmost", "rightmost", "topmost", "bottommost",
    "closest", "nearest", "farthest", "furthest", "closer", "farther",
    "left", "right", "top", "bottom", "upper", "lower", "middle", "center",
    "centre", "camera", "image", "side", "when", "counting", "counted", "one",
    "two", "three", "four", "five", "both", "each", "every",
}


# ---------------------------------------------------------------- 载入
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


# ---------------------------------------------------------------- 基础判定
def is_placeholder(b) -> bool:
    return all(abs(float(b[i]) - FALLBACK[i]) <= TOL for i in range(4))


def np_key(query: str) -> frozenset:
    """物体类别词集合（去掉序数/方位/停用词），用于判断两条 query 是否同类。"""
    s = re.sub(r"[^a-z0-9\s-]", " ", clean_query(query).lower())
    return frozenset(w for w in s.split() if len(w) > 1 and w not in _MOD)


def mutex(q1: str, q2: str) -> bool:
    """两条 query 是否**不可能指向同一目标**（强互斥 -> 组内必有 >=1 条错）。

    判据：同类物体 + 同 kind + 同轴 + 同方向 + 不同 rank。
    反向（left vs right）不算强互斥——同类只有一个实例时二者可共指。
    """
    c1, c2 = parse_spatial_constraint(q1), parse_spatial_constraint(q2)
    if not c1 or not c2:
        return False
    if not (np_key(q1) & np_key(q2)):
        return False
    if c1.get("kind", "pos") != c2.get("kind", "pos"):
        return False
    if c1["axis"] != c2["axis"] or c1["direction"] != c2["direction"]:
        return False
    return int(c1["rank"]) != int(c2["rank"])


# ---------------------------------------------------------------- 主分析
def analyze(pred: dict, qraw: dict) -> dict:
    """把预测拆成「可证明错误」与「分布异常」两组，给出硬下界。"""
    tags = defaultdict(set)

    for qid, b in pred.items():
        if not b or qid not in qraw:
            continue
        x1, y1, x2, y2 = [float(v) for v in b]
        if is_placeholder(b):
            tags[qid].add("placeholder")
            continue
        w, h = x2 - x1, y2 - y1
        if w <= TOL or h <= TOL:
            tags[qid].add("zero_area")
        elif w < THIN or h < THIN:
            tags[qid].add("thin")
        if w * h > HUGE_AREA:
            tags[qid].add("huge")

    by_img = defaultdict(list)
    for qid, b in pred.items():
        if b and qid in qraw:
            by_img[qraw[qid]["visible"]].append(qid)

    shared_excess = 0
    conflict_groups = 0
    conflict_qids = set()
    shared_qids = set()
    conflict_examples = []

    for vis, qs in by_img.items():
        buckets = defaultdict(list)
        for q in qs:
            buckets[tuple(round(float(v), 9) for v in pred[q])].append(q)
        for box, mem in buckets.items():
            if len(mem) < 2:
                continue
            has_conflict = False
            for i in range(len(mem)):
                for j in range(i + 1, len(mem)):
                    if mutex(qraw[mem[i]]["query"], qraw[mem[j]]["query"]):
                        has_conflict = True
            if has_conflict:
                conflict_groups += 1
                conflict_qids.update(mem)
                if len(conflict_examples) < 8:
                    conflict_examples.append({
                        "visible": vis,
                        "box": [round(v, 4) for v in box],
                        "members": [{"id": q, "query": qraw[q]["query"],
                                     "cons": parse_spatial_constraint(qraw[q]["query"])}
                                    for q in mem],
                    })
            free = [m for m in mem
                    if not (tags.get(m, set()) & {"placeholder", "zero_area"})]
            if len(free) >= 2:
                shared_excess += len(free) - 1
                shared_qids.update(free)

    cnt = Counter()
    for qid, ts in tags.items():
        for t in ts:
            cnt[t] += 1
    ph, za = cnt["placeholder"], cnt["zero_area"]
    thin, huge = cnt["thin"], cnt["huge"]

    n = len(qraw)
    strict_floor = ph + za + conflict_groups
    distributional = thin + huge + shared_excess

    strict_qids = ({q for q, ts in tags.items()
                    if ts & {"placeholder", "zero_area"}} | conflict_qids)
    error_qids = {q for q, ts in tags.items() if ts} | conflict_qids | shared_qids
    return {
        "queries": n,
        "predicted": sum(1 for b in pred.values() if b),
        "placeholder": ph,
        "zero_area": za,
        "thin": thin,
        "huge": huge,
        "shared_excess": shared_excess,
        "conflict_groups": conflict_groups,
        "conflict_queries": len(conflict_qids),
        "strict_floor": strict_floor,
        "strict_floor_pct": round(100.0 * strict_floor / n, 2),
        "distributional": distributional,
        "distributional_pct": round(100.0 * distributional / n, 2),
        "panel_total": strict_floor,
        "panel_pct": round(100.0 * strict_floor / n, 2),
        "ceiling_gain": round(strict_floor / float(n), 4),
        "unique_errors": len(error_qids),
        "unique_errors_pct": round(100.0 * len(error_qids) / n, 2),
        "strict_qids": strict_qids,
        "error_qids": error_qids,
        "conflict_examples": conflict_examples,
        "by_image": Counter(qraw[q]["visible"] for q in error_qids),
    }


# ---------------------------------------------------------------- 输出
def print_report(name: str, r: dict, base: dict | None = None) -> None:
    print(f"\n===== {name} =====")
    rows = [
        ("计入", "placeholder  解析失败占位框", r["placeholder"]),
        ("计入", "zero_area    宽或高 <= 1e-6", r["zero_area"]),
        ("计入", "conflict     约束互斥共框组", r["conflict_groups"]),
        ("不计入", "thin         0 < 宽或高 < 0.01", r["thin"]),
        ("不计入", "huge         面积 > 0.5", r["huge"]),
        ("不计入", "shared       同类共框冗余", r["shared_excess"]),
        ("参考", "去重后可修条数（重问队列）", r["unique_errors"]),
    ]
    n = r["queries"]
    if base is None:
        print(f"  {'等级':6s} {'类别':30s} {'条数':>6s} {'占比':>8s}")
        for lvl, label, v in rows:
            print(f"  {lvl:6s} {label:30s} {v:>6d} {100.0 * v / n:>7.2f}%")
    else:
        print(f"  {'类别':30s} {'before':>7s} {'after':>7s} {'delta':>7s}")
        keys = ["placeholder", "zero_area", "conflict_groups",
                "thin", "huge", "shared_excess", "unique_errors"]
        for (lvl, label, _), k in zip(rows, keys):
            a, b = base[k], r[k]
            print(f"  {label:30s} {a:>7d} {b:>7d} {b - a:>+7d}")
    print(f"  {'-' * 56}")
    print(f"  可证明错误面板 {r['panel_total']:>5d}  ({r['panel_pct']}%)"
          f"   -> ACC 硬下界，修复上界 +{r['ceiling_gain']:.4f}")
    print(f"  分布异常合计   {r['distributional']:>5d}  ({r['distributional_pct']}%)"
          f"   不计入面板（thin/huge/shared 经 GT 校准已证伪）")
    print(f"  去重后可修条数 {r['unique_errors']:>5d}  ({r['unique_errors_pct']}%)"
          f"   重问队列规模参考")
    if r["conflict_examples"]:
        print(f"  互斥共框实例（前 {len(r['conflict_examples'])} 组，可直接硬禁）：")
        for ex in r["conflict_examples"]:
            print(f"     {os.path.basename(ex['visible'])}  同框 {ex['box']}")
            for m in ex["members"]:
                print(f"        {m['id']:<14s} {m['query'][:66]}")


def print_gate(ra: dict, rb: dict, pred_a: dict, pred_b: dict) -> None:
    """守门：改动是否「移动了有效框却没说清收益」。

    判定不只看净减少是否 > 0，而看**收益能否背书无依据改动**：
    112 条「有效框 -> 有效框」的改动若只换来 2 条可证明错误修复，则绝大多数
    改动没有验证依据 —— 这正是 09-28 ``chain_enforce=project`` 的形态
    （净减少 +2，官方 ACC -0.0009）。故用「收益背书比 = 净减少 / 无依据改动数」
    分级：>=1.0 才算收益能背书全部改动。

    注意用词：这些改动是**平移（错位）**而非「重排」—— 实测框尺寸 100% 保持、
    位移几乎全在单轴、逐图框多重集全部改变（旧框位置消失、新位置出现）。
    形态判定见 ``scripts/复赛/chain_shift_probe.py``。

    收益口径严格用 ``strict_qids``（placeholder / zero_area / conflict），
    不含 thin / huge / shared —— 后三者已被 GT 校准证伪。
    """
    ea, eb = ra["strict_qids"], rb["strict_qids"]
    fixed, introduced = ea - eb, eb - ea
    moved = [q for q in pred_a if pred_a[q] != pred_b.get(q)]
    moved_clean = [q for q in moved if q not in ea and q not in eb]
    net = len(fixed) - len(introduced)
    ratio = (net / float(len(moved_clean))) if moved_clean else float("inf")

    print("\n----- 改动守门 -----")
    print(f"  predictions 差异条数           {len(moved)}")
    print(f"    其中可证明错误被修好         {len(fixed)}")
    print(f"    其中新引入可证明错误         {len(introduced)}")
    print(f"    其中「有效框 -> 有效框」     {len(moved_clean)}   <- 无验证依据的改动（多为平移错位，非置换）")
    print(f"  可证明错误净减少               {net:+d}")
    if moved_clean:
        print(f"  收益背书比 净减少/无依据改动    {ratio:.3f}"
              f"   （>= 1.0 才算收益能背书全部改动）")
    if not moved:
        print("  判定：两版预测一致。")
    elif net <= 0:
        print("  判定：!! 高风险 !!")
        print("        位移了有效框，但可证明错误净减少 <= 0。")
        print("        没有可验证的收益依据 —— 等同盲改，建议不要上云。")
    elif ratio < 1.0:
        print("  判定：!! 收益不足 !!")
        print(f"        只修好 {net} 条可证明错误，却位移了 {len(moved_clean)} 条有效框。")
        print("        绝大多数改动没有验证依据、风险无界 —— 建议先补证据再上云。")
        print("        （可用 scripts/复赛/chain_shift_probe.py 判定是平移错位还是置换重指派）")
    else:
        print(f"  判定：通过（修好 {len(fixed)} 条，足以背书 {len(moved_clean)} 条无依据改动）。")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="可证明错误面板（无 GT 下的改动守门器）")
    ap.add_argument("--root", default="dataset", help="数据根（含 queries/queries.json）")
    ap.add_argument("-p", "--predictions", required=True, help="A：基线预测")
    ap.add_argument("-c", "--compare", default=None, help="B：对比预测（启用守门）")
    ap.add_argument("--json", default=None, help="导出统计结果")
    ap.add_argument("--top", type=int, default=0, help="列出可证明错误最多的 N 张图")
    args = ap.parse_args()

    qpath = os.path.join(os.path.abspath(args.root), "queries", "queries.json")
    with open(qpath, encoding="utf-8") as fh:
        qraw = json.load(fh)

    pred_a = load_predictions(args.predictions)
    ra = analyze(pred_a, qraw)
    print_report("A: " + args.predictions, ra)

    rb = pred_b = None
    if args.compare:
        pred_b = load_predictions(args.compare)
        rb = analyze(pred_b, qraw)
        print_report("B: " + args.compare, rb, base=ra)
        print_gate(ra, rb, pred_a, pred_b)

    if args.top:
        print(f"\n----- 可证明错误最多的 {args.top} 张图（基线 A） -----")
        for vis, c in ra["by_image"].most_common(args.top):
            print(f"  {c:>3d}  {vis}")

    if args.json:
        def slim(r):
            return {k: v for k, v in r.items()
                    if k not in ("strict_qids", "error_qids", "by_image")}
        payload = {"predictions_a": args.predictions,
                   "predictions_b": args.compare,
                   "metrics_a": slim(ra),
                   "metrics_b": slim(rb) if rb else None}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
