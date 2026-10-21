#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统计提交件的改动规律：IoU 分层、票数分布、改动位置占比。"""
from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = R / "复赛提交材料文件夹" / "参赛团队编号-团队名称-复赛-队长姓名"
DO = PKG / "参赛团队编号-代码与数据" / "data_outputs"
CODE = PKG / "参赛团队编号-代码与数据" / "code27b"


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    d = ar + br - inter
    return inter / d if d > 0 else 0.0


def _bb(d):
    return {k: (v["bbox"] if isinstance(v, dict) else v) for k, v in d.items()}


with zipfile.ZipFile(PKG / "参赛团队编号-赛题结果文档" / "result.zip") as zf:
    sub = _bb(json.loads(zf.read("submission.json").decode("utf-8")))
main = _bb(json.loads((DO / "predictions_27b.json").read_text("utf-8")))
ref = _bb(json.loads((DO / "predictions_ref.json").read_text("utf-8")))
with gzip.open(CODE / "consensus_pool.json.gz", "rt", encoding="utf-8") as fh:
    pool = json.load(fh)

changed = [k for k in sub if sub[k] != main[k]]
ious = {k: _iou(main[k], ref[k]) for k in main}

battle = [k for k, v in ious.items() if v < 0.5]
zero = [k for k, v in ious.items() if v == 0.0]

# 全量分层
def bucket(v):
    if v == 0.0:
        return "0"
    if v < 0.5:
        return "0-0.5"
    if v < 0.9:
        return "0.5-0.9"
    return ">=0.9"


dist = {"0": 0, "0-0.5": 0, "0.5-0.9": 0, ">=0.9": 0}
for v in ious.values():
    dist[bucket(v)] += 1

# 改动位的 IoU 分布
chg_dist = {"0": 0, "0-0.5": 0, "0.5-0.9": 0, ">=0.9": 0}
for k in changed:
    chg_dist[bucket(ious[k])] += 1

# 票数分布（改动位）
votes = []
m = len(pool)
for k in changed:
    sq = 0
    for v in pool.values():
        b = v.get(k)
        if b is None:
            continue
        if _iou(b, ref[k]) > _iou(b, main[k]):
            sq += 1
    votes.append(sq)

stats = {
    "总条目": len(sub),
    "改动": len(changed),
    "战场位(<0.5)": len(battle),
    "完全不相交(0)": len(zero),
    "全量IoU分布": dist,
    "改动位IoU分布": chg_dist,
    "改动位票数": {"满票": sum(1 for v in votes if v == m), "池大小": m},
}
print(json.dumps(stats, ensure_ascii=False, indent=2))
(R / "_viz_stats.json").write_text(
    json.dumps(stats, ensure_ascii=False), "utf-8")
