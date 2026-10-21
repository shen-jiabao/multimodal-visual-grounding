# -*- coding: utf-8 -*-
"""用三模态 GroundingDINO 导出**候选框池**，供"第 k 个"文本推算使用。

动机与背景
- 8B 的 `Detect ALL instances` 在 258 条"空候选"条目上失败（本次重跑 3 种全文提示
  × 4 块分块后仍有 43 条只给 1 框）。
- 用 8B 重跑结果**合并**成候选表的方案已被线上实测否掉：`result_semi_enum_target_E.zip`
  净 −32 条 = **−24.8%**（约 8σ）。失败不在文本推算，而在**候选表本身是垃圾**。
- GroundingDINO 是真正的检测器，其候选池是机制上不同的枚举器；原复赛产物
  `_gd_semi_all.json` 只存了最终框（无候选池），故须重跑。

设计（与远端枚举同纪律：**只采集，取舍在本地**）
- 提示词 = 核心名词（如 `stone`），不是整句问句。
- 存下阈值 0.10 以上的全部框与分数，**不做 NMS、不做排序**——阈值/去重策略改多少次
  都不必再占 GPU。
- 断点粒度 = 一对 (图, core)，每 10 对落盘一次。

用法
    python -B _gd_enum.py --limit 3          # 烟测
    python -B _gd_enum.py                    # 全量目标组（147 对，约 26 min）
    python -B _gd_enum.py --control 40       # 另加 40 对控制组（做召回对照）
"""
from __future__ import annotations
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

import argparse
import json
import sys
import time
from pathlib import Path

R = Path(ROOT_S)
sys.path.insert(0, str(R))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import predict_gd_3modal as G  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
TARGETS = R / "_enum_targets.json"
OUT = R / "_gd_enum_raw.json"
SCORE_FLOOR = 0.10   # 采集下限（远低于判据阈值，留足离线筛选空间）
MAX_BOXES = 60       # 单对上限，防退化输出刷满
MAX_AREA = 0.35      # 与主管线一致：超过视为多目标合并框
CKPT_EVERY = 10


def pairs(targets: dict, q: dict, n_control: int) -> list:
    """待跑 (图, core) 对：目标组全量 + 可选若干控制组。"""
    seen: dict = {}
    for qid, core in targets["target"].items():
        seen.setdefault((q[qid]["visible"], core.lower()), core)
    ctl = sorted(targets["control"].items())[:n_control]
    for qid, core in ctl:
        seen.setdefault((q[qid]["visible"], core.lower()), core)
    return sorted(seen.items())


def run_pair(model, visible: str, core: str) -> dict:
    """跑一对，返回采集记录。"""
    root = SEMI
    src, mm, h, w, _depth = G.load_multimodal_image(
        str(root / visible),
        str(root / visible.replace("/visible/", "/infrared/")),
        str(root / visible.replace("/visible/", "/depth/")))
    boxes, logits = G.predict_multimodal(model, mm, core)
    xyxy = G.to_xyxy_norm(boxes, h, w)
    out_b, out_s = [], []
    for b, s in zip(xyxy, logits.tolist()):
        if s < SCORE_FLOOR:
            continue
        nb = G.sanitize_box(b)
        if (nb[2] - nb[0]) * (nb[3] - nb[1]) > MAX_AREA:
            continue
        out_b.append([round(float(v), 5) for v in nb])
        out_s.append(round(float(s), 5))
    order = np.argsort([-s for s in out_s])[:MAX_BOXES]
    return {"visible": visible, "core": core,
            "bbox_all": [out_b[i] for i in order],
            "scores": [out_s[i] for i in order]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对（烟测）")
    ap.add_argument("--control", type=int, default=0, help="额外跑 N 对控制组")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    tg = json.loads(TARGETS.read_text(encoding="utf-8"))
    todo = pairs(tg, q, a.control)
    if a.limit:
        todo = todo[:a.limit]
    out_path = Path(a.out)
    done: dict = {}
    if out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8"))
    print(f"待跑 {len(todo)} 对 | 已有 {len(done)} 对", flush=True)

    print("载入三模态模型 …", flush=True)
    t0 = time.time()
    model = G.load_model_3modal()
    print(f"  模型就绪，用时 {time.time() - t0:.1f} s", flush=True)

    t0, n_new = time.time(), 0
    for i, ((visible, core), _raw) in enumerate(todo, 1):
        key = f"{visible}||{core}"
        if key in done:
            continue
        try:
            rec = run_pair(model, visible, core)
        except Exception as exc:                       # noqa: BLE001
            rec = {"visible": visible, "core": core, "bbox_all": [],
                   "scores": [], "err": str(exc)[:120]}
        done[key] = rec
        n_new += 1
        if n_new % CKPT_EVERY == 0:
            out_path.write_text(json.dumps(done, ensure_ascii=False),
                                encoding="utf-8")
        if i % 5 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  [{i}/{len(todo)}] {el/max(n_new,1):.1f} s/对 "
                  f"| 本对 n={len(rec['bbox_all'])} | 剩余 "
                  f"约 {(len(todo)-i)*el/max(n_new,1)/60:.0f} min", flush=True)
    out_path.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
    ns = [len(v["bbox_all"]) for v in done.values()]
    print(f"\n完成：{len(done)} 对 → {out_path.name}")
    if ns:
        print(f"  候选数：≥2 的 {sum(1 for n in ns if n >= 2)}/{len(ns)}"
              f" | 均值 {sum(ns)/len(ns):.1f} | ≥10 框 {sum(1 for n in ns if n >= 10)}")
    if torch.cuda.is_available():
        print("  峰值显存 %.0f MiB" % (torch.cuda.max_memory_allocated() / 2**20))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
