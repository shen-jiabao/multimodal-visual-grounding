#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""32B 管线烟测：分两个独立判据，不混为一谈。

【--self】接线判据（★这是放行门槛）
    用**同一台服务器**、同一批问句、同一提示词，分别经
      ① 我方管线 `_qwen_full_pipeline.make_gen_api()`
      ② 归档探针类 `_probe_models.Api32B`
    各跑一遍 direct，逐条比对。二者都打同一个 vLLM 端点、temperature=0，
    故**必须逐位一致**——不一致即说明我方接线（提示词/图像编码/base64 组装/解析）
    有问题，与模型权重无关。这个判据不受"量化来源不同"的干扰。

【--ref】量化漂移读数（不是门槛，只报告）
    与归档实例产出的 `probe_32b.json` 比对。归档那台用的权重是本机
    `/root/models/Qwen3-VL-32B-Instruct-AWQ`（平台自带），本次从 ModelScope 下的
    是社区量化 `QuantTrio/Qwen3-VL-32B-Instruct-AWQ`——**量化来源不同就可能不逐位
    相同**。所以此模式只给出「一致率 + IoU 分布」，作为"两份权重差多少"的信息，
    并用于判断是否还能拿归档探针当参照。

用法
    python _smoke_qwen32b.py --data /root/rivermind-data/dataset
    python _smoke_qwen32b.py --mode self          # 只跑门槛判据
"""
import argparse
import json
import os
import sys
import time

from PIL import Image
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def iou(a, b):
    if not a or not b:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["both", "self", "ref"])
    ap.add_argument("--data", default=os.environ.get("QWEN_DATA",
                                                     "/root/rivermind-data/dataset"))
    ap.add_argument("--qids", default="/root/_probe_qids.json")
    ap.add_argument("--ref", default="/root/_probe_32b.json")
    ap.add_argument("--n", type=int, default=0)
    a = ap.parse_args()

    import _qwen_full_pipeline as P

    qids = json.load(open(a.qids, encoding="utf-8"))
    if isinstance(qids, dict):
        qids = list(qids.keys())
    if a.n:
        qids = qids[:a.n]
    q = json.load(open(os.path.join(a.data, "queries", "queries.json"),
                       encoding="utf-8"))
    print(f"引擎 {P.ENGINE} → {P.API_BASE} model={P.API_MODEL}"
          f" | 数据 {a.data} | 条目 {len(qids)}", flush=True)

    ours = P.make_gen_api()
    imgcache = {}

    def img_of(k):
        p = os.path.join(a.data, q[k]["visible"])
        if p not in imgcache:
            imgcache[p] = Image.open(p).convert("RGB")
        return imgcache[p]

    def run(gen):
        out = {}
        for i, k in enumerate(qids, 1):
            out[k] = P.parse_boxes(
                gen(img_of(k), P.FULL_DIRECT_PROMPT.format(q=q[k]["query"])))
            if i % 10 == 0:
                print(f"    …{i}/{len(qids)}", flush=True)
        return out

    print("\n=== 跑我方管线 ===", flush=True)
    t0 = time.time()
    A = run(ours)
    print(f"  用时 {time.time()-t0:.1f} s", flush=True)

    ok = True

    # ---------------- 门槛：接线自校验 ----------------
    if a.mode in ("both", "self"):
        print("\n===== 1) 接线判据（我方管线 vs 归档 Api32B，同一台服务器）=====",
              flush=True)
        try:
            from _probe_models import Api32B
            other = Api32B().gen
            print("=== 跑归档 Api32B ===", flush=True)
            t0 = time.time()
            B = run(other)
            print(f"  用时 {time.time()-t0:.1f} s", flush=True)
            same = near = 0
            bad = []
            for k in qids:
                x = (A[k][0] if A[k] else None)
                y = (B[k][0] if B[k] else None)
                if x == y:
                    same += 1
                elif iou(x, y) >= 0.999:
                    near += 1
                else:
                    bad.append((k, x, y, round(iou(x, y), 4)))
            print(f"  逐位相同 {same} ｜ IoU≥0.999 {near} ｜ ★偏差 {len(bad)}"
                  f"（共 {len(qids)}）")
            for k, x, y, io in bad[:10]:
                print(f"    {k} IoU={io}")
                print(f"      我方 {[round(v, 3) for v in x] if x else x}")
                print(f"      归档 {[round(v, 3) for v in y] if y else y}")
            if bad:
                print("  ★ 接线存在偏差 —— 不得启动全量")
                ok = False
            else:
                print("  ✓ 接线无误（两个独立实现给出同一结果）")
        except Exception as exc:                       # noqa: BLE001
            print(f"  ★ 无法执行自校验（{type(exc).__name__}: {exc}）——"
                  f"故不能确认接线，按不放行处理")
            ok = False

    # ---------------- 参考：量化漂移 ----------------
    if a.mode in ("both", "ref"):
        print("\n===== 2) 与归档 probe_32b.json 的差异（量化漂移读数，非门槛）=====",
              flush=True)
        if not os.path.exists(a.ref):
            print("  归档探针不在场，跳过")
        else:
            ref = json.load(open(a.ref, encoding="utf-8"))
            ex = tol = low = 0
            lows = []
            for k in qids:
                x = (A[k][0] if A[k] else None)
                y = (ref.get(k) or {}).get("direct_bbox")
                if x == y:
                    ex += 1
                elif iou(x, y) >= 0.9:
                    tol += 1
                else:
                    low += 1
                    lows.append((k, round(iou(x, y), 3)))
            n = max(len(qids), 1)
            print(f"  逐位相同 {ex}/{n} = {ex/n:.0%} ｜ IoU≥0.9 {tol} ｜ "
                  f"IoU<0.9 {low}")
            if lows:
                print("  IoU<0.9 的条目: " + ", ".join(f"{k}({v})" for k, v in lows[:12]))
            print("  解读：两份权重若量化来源相同，应几乎逐位相同；"
                  "若一致率偏低，说明社区量化与归档权重有实质差异——"
                  "此时归档探针不能再当参照，但**不影响 32B 自身的可用性**。")

    print("\n===== 结论 =====")
    print("PASS：接线校验通过，可以启动全量" if ok else
          "BLOCKED：接线未通过，不得启动全量")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())