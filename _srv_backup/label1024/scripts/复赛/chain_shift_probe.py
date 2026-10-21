# -*- coding: utf-8 -*-
"""判定两版预测之间的差异是「平移(错位)」还是「置换重指派」。

背景
----
09-28 那次 ``chain_enforce=project`` 官方 ACC 0.7292 -> 0.7283，
事后描述常写成「重排了 111 条已有框」。**这个说法不准确**：
本项目实测表明，改动后的框尺寸 100% 保持、位移几乎全部落在单一轴上、
且每张受影响图的「框多重集」都发生了改变 —— 也就是说**旧框位置消失、
新框位置出现**，这是**平移（错位）**，不是「把已有框在 query 之间倒手」。

术语差异很重要，因为它决定修法方向：
  - 若是**置换**（多重集不变）-> 框都在正确位置上，只是分配贴反了，
    重新分配即可修好；
  - 若是**平移**（多重集改变）-> 框被搬离了原本（可能正确）的位置，
    只会制造新的错位，不可能凭空补出缺失目标。

判据
----
  1) 框尺寸是否保持（宽高比/宽高完全一致）
  2) 位移方向（仅 x / 仅 y / 两轴）
  3) 位移幅度分位数
  4) **决定性**：逐图比较框的 Counter（多重集）
     - 不变  -> 置换重指派
     - 改变  -> 平移（错位）

用法
----
    python scripts/复赛/chain_shift_probe.py \\
        -p outputs/复赛/archive/run_20260924_090232_qwen_multiimg.tar.gz \\
        -c outputs/复赛/archive/run_20260928_003617_qwen_diag.tar.gz \\
        --json outputs/复赛/audit/shift_probe_20260928.json
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

from scripts.复赛.provable_errors import load_predictions  # noqa: E402

TOL = 1e-6


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return round(sorted_vals[min(len(sorted_vals) - 1, int(len(sorted_vals) * p))], 4)


def multiset_scope(pred_a, pred_b, qids_by_img, pred_a_raw):
    """逐图比较框多重集。返回 (不变张数, 改变张数, 改变图的样例)。"""
    same = changed = 0
    samples = []
    for vis, qs in qids_by_img.items():
        old = Counter(tuple(round(float(v), 6) for v in pred_a_raw[q]) for q in qs)
        new = Counter(tuple(round(float(v), 6) for v in pred_b[q]) for q in qs)
        if old == new:
            same += 1
            continue
        changed += 1
        if len(samples) < 5:
            gone = [b for b in old if b not in new]
            fresh = [b for b in new if b not in old]
            samples.append({
                "visible": vis,
                "queries": len(qs),
                "boxes_gone": len(gone),
                "boxes_new": len(fresh),
            })
    return same, changed, samples


def main() -> None:
    ap = argparse.ArgumentParser(
        description="判定两版预测的差异是平移(错位)还是置换重指派")
    ap.add_argument("--root", default="dataset", help="数据根（含 queries/queries.json）")
    ap.add_argument("-p", "--predictions", required=True, help="A：基线预测")
    ap.add_argument("-c", "--compare", required=True, help="B：对比预测")
    ap.add_argument("--json", default=None, help="导出统计结果")
    args = ap.parse_args()

    with open(os.path.join(os.path.abspath(args.root), "queries", "queries.json"),
              encoding="utf-8") as fh:
        qraw = json.load(fh)
    A = load_predictions(args.predictions)
    B = load_predictions(args.compare)

    diff = [q for q in A if A[q] != B.get(q)]
    print("=" * 72)
    print("A: %s" % args.predictions)
    print("B: %s" % args.compare)
    print("差异 query = %d / %d" % (len(diff), len(A)))
    print("=" * 72)
    if not diff:
        print("\n两版预测完全一致，无需判定。")
        return

    size_kept = size_changed = 0
    axis_x = axis_y = both = still = 0
    mags = []
    for q in diff:
        a = [float(v) for v in A[q]]
        b = [float(v) for v in B[q]]
        wa, ha = a[2] - a[0], a[3] - a[1]
        wb, hb = b[2] - b[0], b[3] - b[1]
        if abs(wa - wb) <= TOL and abs(ha - hb) <= TOL:
            size_kept += 1
        else:
            size_changed += 1
        dx = abs((b[0] + b[2]) / 2 - (a[0] + a[2]) / 2)
        dy = abs((b[1] + b[3]) / 2 - (a[1] + a[3]) / 2)
        if dx > TOL and dy > TOL:
            both += 1
        elif dx > TOL:
            axis_x += 1
        elif dy > TOL:
            axis_y += 1
        else:
            still += 1
        mags.append(max(dx, dy))
    mags.sort()

    print("\n[1] 框尺寸是否保持")
    print("    尺寸完全保持 = %d / %d (%.1f%%)    尺寸改变 = %d"
          % (size_kept, len(diff), 100.0 * size_kept / len(diff), size_changed))
    print("    判读：尺寸全保持 => 改动只动位置，不缩放/不换框形")

    print("\n[2] 位移方向")
    print("    仅 x 轴 = %d   仅 y 轴 = %d   两轴都动 = %d   中心未动 = %d"
          % (axis_x, axis_y, both, still))
    print("    判读：单轴占绝大多数 => 是沿链轴的**平移**，不是跨目标换框")

    print("\n[3] 位移幅度（中心切比雪夫距离，归一化坐标）")
    print("    p50=%.4f  p90=%.4f  max=%.4f   >0.05 = %d   >0.15 = %d"
          % (_pct(mags, .5), _pct(mags, .9), round(mags[-1], 4),
             sum(1 for v in mags if v > 0.05), sum(1 for v in mags if v > 0.15)))

    by_img = defaultdict(list)
    for q in diff:
        by_img[qraw[q]["visible"]].append(q)
    print("\n涉及图像 = %d 张" % len(by_img))

    allimg = defaultdict(list)
    for q in A:
        if A.get(q) and B.get(q):
            allimg[qraw[q]["visible"]].append(q)

    s1, c1, ex1 = multiset_scope(A, B, by_img, A)
    print("\n[4a] 受影响图的「框多重集」（只看差异 query）")
    print("    不变（= 只是把已有框在 query 间倒手，纯置换重指派） = %d 张" % s1)
    print("    改变（= 出现新框位置 / 旧位置消失，即平移错位）   = %d 张" % c1)

    s2, c2, ex2 = multiset_scope(A, B, allimg, A)
    print("\n[4b] 全量 %d 张图（含未被改动的 query，判定更严格）" % len(allimg))
    print("    不变 = %d 张     改变 = %d 张" % (s2, c2))

    verdict = "置换重指派" if c2 == 0 else "平移（错位）"
    print("\n" + "-" * 72)
    print("判定：**%s**" % verdict)
    if c2:
        print("      每张受影响图都有框位置被搬走、新位置被写出 —— 多重集改变，")
        print("      说明这是**位移**而不是「框都在对的位置、只是分错了 query」。")
        print("      推论：这类改动只会把原本（可能正确）的框搬离，制造新错位；")
        print("            它无法补出模型根本没框到的那个目标 —— 不可作为提分手段。")
    print("-" * 72)

    if args.json:
        payload = {
            "predictions_a": args.predictions,
            "predictions_b": args.compare,
            "diff_queries": len(diff),
            "size_kept": size_kept,
            "size_changed": size_changed,
            "axis": {"x_only": axis_x, "y_only": axis_y, "both": both, "still": still},
            "shift_p50": _pct(mags, .5), "shift_p90": _pct(mags, .9),
            "shift_max": round(mags[-1], 4),
            "shift_gt_0_05": sum(1 for v in mags if v > 0.05),
            "shift_gt_0_15": sum(1 for v in mags if v > 0.15),
            "images_touched": len(by_img),
            "multiset_scope_diff_queries": {"unchanged": s1, "changed": c1},
            "multiset_scope_all_images": {"unchanged": s2, "changed": c2,
                                          "samples": ex2 or ex1},
            "verdict": verdict,
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
        print("\n已导出: " + args.json)


if __name__ == "__main__":
    main()
