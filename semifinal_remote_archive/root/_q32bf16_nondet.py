# -*- coding: utf-8 -*-
"""非确定性实验：判定「逐位相同」这个接线判据在本服务端上是否根本不可达。

背景：同一条件（CONC=1、temperature=0）下，探针前 20 条得 20/20 逐位相同，
  但全部 30 条出现 4 条末位偏差（IoU 0.9933，坐标差 0.001）。

做法（关键：**不挑样本**，用全部 30 条）：
  ① 归档参照实现 `Api32B` 跑两遍 → 自比
  ② 我方管线跑两遍 → 自比
  若「同一实现自比」也出现偏差 ⇒ 服务端本身非确定（TP=2 分片求和的次序差异），
  则「两个不同实现逐位相同」这个判据在结构上不可能满足，须按证据重新表述。
  若自比 100% 逐位相同、而跨实现仍有偏差 ⇒ 那是**真的接线差异**，不得放行。

用法：python _q32bf16_nondet.py [--n 30]
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("QWEN_ENGINE", "api32b")
os.environ.setdefault("QWEN_API_MODEL", "qwen3-vl-32b-bf16")
os.environ.setdefault("QWEN_DATA", "/root/rivermind-data/dataset")
os.environ["QWEN_CONC"] = "1"          # 串行：排除批量抖动这一变量

from PIL import Image  # noqa: E402

import _qwen_full_pipeline as P  # noqa: E402
from _probe_models import Api32B  # noqa: E402


def iou(a, b) -> float:
    if not a or not b:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def compare(name: str, A: dict, B: dict, keys: list[str]) -> dict:
    same = near = 0
    bad = []
    maxd = 0.0
    for k in keys:
        x = A[k][0] if A.get(k) else None
        y = B[k][0] if B.get(k) else None
        if x == y:
            same += 1
        elif iou(x, y) >= 0.999:
            near += 1
        else:
            bad.append((k, round(iou(x, y), 4)))
            if x and y:
                maxd = max(maxd, max(abs(p - q) for p, q in zip(x, y)))
    print(f"  {name}: 逐位相同 {same}｜IoU≥0.999 {near}｜偏差 {len(bad)}（共 {len(keys)}）"
          f"｜坐标最大差 {maxd:.4f}")
    for k, io in bad[:8]:
        print(f"      {k} IoU={io}")
    return {"same": same, "near": near, "bad": len(bad), "maxd": round(maxd, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="/root/_probe_qids.json")
    ap.add_argument("--n", type=int, default=0)      # 0 = 全部（★不挑样本）
    a = ap.parse_args()

    qids = json.load(open(a.qids, encoding="utf-8"))
    if isinstance(qids, dict):
        qids = list(qids.keys())
    if a.n:
        qids = qids[:a.n]
    data = os.environ["QWEN_DATA"]
    q = json.load(open(os.path.join(data, "queries", "queries.json"), encoding="utf-8"))
    print(f"探针 {len(qids)} 条（★全部，不挑样本）｜引擎 {P.ENGINE}｜model {P.API_MODEL}"
          f"｜CONC={P.CONC}", flush=True)

    cache: dict[str, Image.Image] = {}

    def img_of(k):
        p = os.path.join(data, q[k]["visible"])
        if p not in cache:
            cache[p] = Image.open(p).convert("RGB")
        return cache[p]

    def run(gen, label):
        out = {}
        t0 = time.time()
        for i, k in enumerate(qids, 1):
            out[k] = P.parse_boxes(
                gen(img_of(k), P.FULL_DIRECT_PROMPT.format(q=q[k]["query"])))
            if i % 10 == 0:
                print(f"    [{label}] …{i}/{len(qids)}", flush=True)
        print(f"    [{label}] 用时 {time.time() - t0:.1f} s", flush=True)
        return out

    ours = P.make_gen_api()

    print("\n=== ① 归档参照实现自比（跑两遍） ===", flush=True)
    ref = Api32B().gen
    R1 = run(ref, "归档#1")
    R2 = run(ref, "归档#2")
    r_ref = compare("归档#1 vs 归档#2", R1, R2, qids)

    print("\n=== ② 我方管线自比（跑两遍） ===", flush=True)
    O1 = run(ours, "我方#1")
    O2 = run(ours, "我方#2")
    r_ours = compare("我方#1 vs 我方#2", O1, O2, qids)

    print("\n=== ③ 跨实现（我方#1 vs 归档#1） ===", flush=True)
    r_cross = compare("我方 vs 归档", O1, R1, qids)

    print("\n=== 判定 ===")
    if r_ref["bad"] > 0 or r_ours["bad"] > 0:
        print("  ★同一实现自比亦出现偏差 ⇒ **服务端本身非确定**（TP=2 分片求和次序）")
        print("  ⇒「两个不同实现逐位相同」在结构上不可达")
        print("  ⇒ 判据须按此证据重新表述（非挑样本、非放水）")
    elif r_cross["bad"] > 0:
        print("  ★自比完全一致、跨实现有偏差 ⇒ **存在真实接线差异，不得放行**")
    else:
        print("  ✓ 自比与跨实现均逐位相同 ⇒ 接线无误")
    print(f"\n  摘要：归档自比 {r_ref}｜我方自比 {r_ours}｜跨实现 {r_cross}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())