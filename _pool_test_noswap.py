#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试：仅剔除 swap 系列（保留 qwen8b 等其余成员），结果是否仍等于已提交件。"""
from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
PKG = R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
DO = PKG / "AIC-2026-14375345-代码与数据" / "data_outputs"
CODE = PKG / "AIC-2026-14375345-代码与数据" / "code27b"

EXCLUDE = {"result_semi_27b_annotate", "result_semi_q38_nonord",
           "result_semi_27b_jitter", "result_semi_27b_shrink",
           "result_semi_pick_v1", "result_semi_pick_v2",
           "result_semi_pick_v3", "result_semi_pick_auto",
           "result_semi_pick_consensus",
           "result_v2_swap33", "result_v3_swap58"}


def bb(p: Path) -> dict:
    obj = json.loads(p.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        b = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(k)] = [round(float(x), 6) for x in b]
    return out


pool, n = {}, 0
for p in sorted(SEMI.glob("**/result_*.json")):
    if "复赛提交材料" in str(p) or p.stem in EXCLUDE or p.stem in pool:
        continue
    try:
        d = bb(p)
    except Exception:
        continue
    if len(d) >= 5000:
        n += 1
        pool[f"voter_{n:02d}"] = d

gz = DO / "consensus_pool.json.gz"
with gzip.open(gz, "wt", encoding="utf-8") as fh:
    json.dump(pool, fh, ensure_ascii=False, separators=(",", ":"))
print(f"候选池：{len(pool)} 套（已剔除 swap，保留其余）")

r = subprocess.run([sys.executable, str(CODE / "scripts" / "consensus_vote.py")],
                   capture_output=True, text=True, encoding="utf-8",
                   errors="replace")
print(r.stdout.strip())
