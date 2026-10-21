#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""预测 JSON → submission.json（坐标 round 4 位；缺失/非法回退到占位框）。

用法
----
    python make_sub.py --queries <queries.json> \\
                       --predictions <predictions.json> \\
                       --output <submission.json>

说明
----
* 预测 JSON 支持两种形态：``{qid: [x1,y1,x2,y2]}`` 或 ``{qid: {"bbox": [...]}}``；
* 输出为平台提交格式 ``{qid: {visible, infrared, depth, query, bbox}}``——
  除 bbox 外的四个字段原样回显，不做任何改动（赛题要求）；
* bbox 统一 round 至 4 位小数，并做合法性检查（有序、落在 [0,1]），
  不合法或缺失时回退到占位框。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FALLBACK = [0.49, 0.49, 0.51, 0.51]


def main() -> int:
    ap = argparse.ArgumentParser(description="预测 JSON → submission.json")
    ap.add_argument("--queries", required=True, help="queries.json 路径")
    ap.add_argument("--predictions", required=True, help="预测 JSON 路径")
    ap.add_argument("--output", required=True, help="输出 submission.json 路径")
    ap.add_argument("--fallback", nargs=4, type=float, default=FALLBACK,
                    help="缺失/非法预测的回退框（默认 0.49 0.49 0.51 0.51）")
    a = ap.parse_args()

    queries = json.loads(Path(a.queries).read_text(encoding="utf-8"))
    raw = json.loads(Path(a.predictions).read_text(encoding="utf-8"))

    def _bbox(v) -> list[float]:
        """取框 → round 4 位 → 合法性检查；异常回退占位框。"""
        b = v.get("bbox") if isinstance(v, dict) else v
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            return list(a.fallback)
        b = [round(float(x), 4) for x in b]
        if not (0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            return list(a.fallback)
        return b

    out = {}
    for k, q in queries.items():
        # 除 bbox 外的字段原样回显，保证评测程序能正确匹配
        out[k] = {
            "visible": q["visible"],
            "infrared": q["infrared"],
            "depth": q["depth"],
            "query": q["query"],
            "bbox": _bbox(raw.get(k)),
        }

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"submission: {len(out)} entries → {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
