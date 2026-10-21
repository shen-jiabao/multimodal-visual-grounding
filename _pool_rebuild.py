#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重建纯模型输出的投票池：剔除派生成员（swap 等）与完全重复成员。

剔除规则
  1. 文件名标记为派生版本的（swap / pick / jitter / shrink 等）一律不进池；
  2. 内容与其他成员完全相同者只保留一个（去重）；
  3. 顺带报告池内两两最小差异，暴露潜在"近重复"。
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
PKG = R / "复赛提交材料文件夹" / "AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝"
DO = PKG / "AIC-2026-14375345-代码与数据" / "data_outputs"
CODE = PKG / "AIC-2026-14375345-代码与数据" / "code27b"

# 派生/非独立产物：主件、参照件、以及一切"从别的产物改出来的"版本
DERIVED = {
    "result_semi_27b_annotate", "result_semi_q38_nonord",
    "result_semi_27b_jitter", "result_semi_27b_shrink",
    "result_semi_pick_v1", "result_semi_pick_v2", "result_semi_pick_v3",
    "result_semi_pick_auto", "result_semi_pick_consensus",
    "result_v2_swap33", "result_v3_swap58",
}
KEYWORDS = ("swap", "pick", "jitter", "shrink")


def _bboxes(p: Path) -> dict:
    obj = json.loads(p.read_text("utf-8"))
    out = {}
    for k, v in obj.items():
        b = v.get("bbox") if isinstance(v, dict) else v
        if isinstance(b, (list, tuple)) and len(b) == 4:
            out[str(k)] = [round(float(x), 6) for x in b]
    return out


def main() -> int:
    pool, seen_hash, dropped = {}, {}, []
    for p in sorted(SEMI.glob("**/result_*.json")):
        if "复赛提交材料" in str(p):
            continue
        name = p.stem
        if name in DERIVED or any(k in name for k in KEYWORDS):
            dropped.append((name, "派生/非独立"))
            continue
        if name in pool:
            continue
        try:
            d = _bboxes(p)
        except Exception:
            continue
        if len(d) < 5000:
            continue
        h = hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
        if h in seen_hash:
            dropped.append((name, f"与 {seen_hash[h]} 内容完全相同"))
            continue
        seen_hash[h] = name
        pool[f"voter_{len(pool)+1:02d}"] = d

    gz = DO / "consensus_pool.json.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"重建后投票池：{len(pool)} 套 → {gz.name}（{gz.stat().st_size/1e6:.2f} MB）")
    print("\n剔除的成员：")
    for n, why in dropped:
        print(f"  {n:<45} {why}")

    # 池内两两最小差异（找近重复）
    names = list(pool)
    print("\n池内最相似的三对（差异条数）：")
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = pool[names[i]], pool[names[j]]
            common = [k for k in a if k in b]
            pairs.append((sum(1 for k in common if a[k] != b[k]), names[i], names[j]))
    pairs.sort()
    for cnt, a, b in pairs[:3]:
        print(f"  {cnt:>5} 条  {a} ↔ {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
