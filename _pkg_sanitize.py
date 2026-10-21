#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""材料包净化：重建中性 key 的投票池（消除实验代号），并备好包内参照件。"""
from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
SUB = SEMI / "未提交"
PKG = R / "复赛提交材料文件夹" / "参赛团队编号-团队名称-复赛-队长姓名"
DO = PKG / "参赛团队编号-代码与数据" / "data_outputs"
CODE = PKG / "参赛团队编号-代码与数据" / "code27b"

SELF = {"result_semi_27b_annotate", "result_semi_q38_nonord",
        "result_semi_27b_jitter", "result_semi_27b_shrink",
        "result_semi_pick_v1", "result_semi_pick_v2", "result_semi_pick_v3",
        "result_semi_pick_auto", "result_semi_pick_consensus",
        "predictions_27b", "predictions_ref", "predictions_consensus"}


def _bboxes(path: Path) -> dict:
    obj = json.loads(path.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        bb = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(bb, (list, tuple)) and len(bb) == 4:
            out[str(k)] = [round(float(x), 6) for x in bb]
    return out


def main() -> int:
    files = sorted(SEMI.glob("**/result_*.json"))
    pool, n = {}, 0
    for p in files:
        if "复赛提交材料" in str(p) or p.stem in SELF:
            continue
        if p.stem in pool:
            continue
        try:
            d = _bboxes(p)
        except Exception:
            continue
        if len(d) >= 5000:
            n += 1
            pool[f"voter_{n:02d}"] = d

    gz = CODE / "consensus_pool.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"投票池 {len(pool)} 套（中性命名 voter_01..voter_{len(pool):02d}）"
          f" → {gz.name}（{gz.stat().st_size/1e6:.2f} MB）")

    # 参照件入包（管线输入之一）
    ref_src = SEMI / "0.7963" / "result_semi_q38_nonord.json"
    shutil.copy(ref_src, DO / "predictions_ref.json")
    print(f"参照件 → data_outputs/predictions_ref.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
