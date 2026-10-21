#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多产物共识投票后处理：由 27B 主件 + 参照件 + 投票池自动产出提交包。

规则
----
对每个 query，令投票池中「更接近参照件」的票数为 sq、「更接近主件」的票数为 sa：

  1. 仅当该位属于主件与参照件的**真分歧位**（两框互 IoU < 0.5）时，才进入投票判定；
     两框互 IoU ≥ 0.5 的同侧位置不构成有效改动，直接跳过。
  2. ``sa == 0 且 sq == len(pool)``（主件零票、参照件满票）→ 采用参照件框。
  3. 其余位置一律保持主件框。

阈值取「满票」：该档由投票池全体成员的一致意见支撑，可避免引入低一致性扰动。

输入（相对本脚本）
  ../data_outputs/predictions_27b.json   主件：27B 全量推理结果
  ../data_outputs/predictions_ref.json   参照件：另一次解码结果，用作比对基准
  ../data_outputs/consensus_pool.json.gz  投票池：多套产物坐标

输出
  ../data_outputs/predictions_consensus.json
  ../data_outputs/result.zip             提交包（内部 submission.json；时间戳固定，确定性）

用法
  python scripts/consensus_vote.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

# 工程根：默认取本脚本上两级（code27b/），其同级目录含 data_outputs/
_ap = argparse.ArgumentParser(description="多产物共识投票后处理")
_ap.add_argument("--code-root", default=None,
                 help="可选：code27b 目录路径（默认按脚本位置推断）")
_a = _ap.parse_args()

SCRIPTS = Path(__file__).resolve().parent
CODE = Path(_a.code_root).resolve() if _a.code_root else SCRIPTS.parent
DATA = CODE.parent / "data_outputs"
POOL_PATH = DATA / "consensus_pool.json.gz"

MAIN = DATA / "predictions_27b.json"
REF = DATA / "predictions_ref.json"
OUT_JSON = DATA / "predictions_consensus.json"
OUT_ZIP = DATA / "result.zip"


def _iou(a, b) -> float:
    """两个归一化框 [x1,y1,x2,y2] 的交并比。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    d = ar + br - inter
    return inter / d if d > 0 else 0.0


def _bboxes(path: Path) -> dict:
    """读预测 JSON，取 {qid: [x1,y1,x2,y2]}。"""
    obj = json.loads(path.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        bb = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            out[str(k)] = [float(x) for x in bb]
    return out


def main() -> int:
    cur = json.loads(MAIN.read_text("utf-8"))
    main_bb = _bboxes(MAIN)
    ref_bb = _bboxes(REF)
    with gzip.open(POOL_PATH, "rt", encoding="utf-8") as fh:
        pool = json.load(fh)
    m = len(pool)

    out = {k: dict(v) for k, v in cur.items()}
    n_adopted = 0
    for k in main_bb:
        if k not in ref_bb:
            continue
        # 规则 1：仅真分歧位进入投票判定
        if _iou(main_bb[k], ref_bb[k]) >= 0.5:
            continue
        sa = sq = 0
        for bboxes in pool.values():
            b = bboxes.get(k)
            if b is None:
                continue
            ia, iq = _iou(b, main_bb[k]), _iou(b, ref_bb[k])
            if ia > iq:
                sa += 1
            elif iq > ia:
                sq += 1
        # 规则 2：主件零票、参照件满票 → 采纳参照件
        if sa == 0 and sq == m:
            out[k]["bbox"] = list(ref_bb[k])
            n_adopted += 1

    # 格式自检
    for k, e in out.items():
        b = e["bbox"]
        assert len(b) == 4 and all(isinstance(x, (int, float)) for x in b), k
        assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1, (k, b)

    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    # 固定 zip 内部时间戳 → 同输入必得同 SHA256
    zi = zipfile.ZipInfo("submission.json", date_time=(2026, 10, 5, 0, 0, 0))
    zi.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zi, OUT_JSON.read_text("utf-8"))

    sha = hashlib.sha256(OUT_ZIP.read_bytes()).hexdigest().upper()
    fixed = sum(1 for k in out if out[k]["bbox"] != cur[k]["bbox"])
    print(f"投票池 {m} 套｜满票位修正 {n_adopted} 条｜相对主件修正 {fixed} 条")
    print(f"输出 {OUT_ZIP.name}｜SHA256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
