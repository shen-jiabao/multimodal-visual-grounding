# -*- coding: utf-8 -*-
"""复赛提交件格式校验（独立于融合器）。

刻意**不复用** `_fuse_semi` 的内存对象，而是直接读最终要提交的 zip：
读盘产物才是真正提交的东西，自己校自己不算校验。
判据（bbox 是否有序且落在 [0,1]）复用 `_fuse_qwen.ordered`，不另写一份。

用法：
    python _validate_semi.py 结果.zip|结果.json [--queries 复赛queries.json]
退出码：0 通过；1 有硬性问题。
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

from _fuse_qwen import ordered  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
FIELDS = {"bbox", "depth", "infrared", "query", "visible"}


def load(path: Path) -> tuple[dict, str]:
    """从 zip 或 json 读提交数据，返回 (数据, 来源描述)。"""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            if len(names) != 1:
                print(f"  [警告] zip 内含 {len(names)} 个文件: {names}")
            data = json.loads(zf.read(names[0]).decode("utf-8"))
            return data, f"{path.name} 内部 {names[0]}"
    return json.loads(path.read_text(encoding="utf-8")), path.name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="待校验的 zip 或 json")
    ap.add_argument("--queries", default=str(SEMI / "queries" / "queries.json"))
    a = ap.parse_args()

    sp, qp = Path(a.submission), Path(a.queries)
    if not sp.exists():
        print(f"找不到 {sp}")
        return 1
    data, src = load(sp)
    queries = json.loads(qp.read_text(encoding="utf-8"))
    print(f"提交件: {src}")
    print(f"queries: {qp.name}")

    hard = []          # 硬性问题 → 退出码 1
    soft = []          # 需你决策/知悉，不判失败

    print(f"\n[1] 条数  {len(data)} / 应 {len(queries)}", end="  ")
    if len(data) == len(queries):
        print("OK")
    else:
        print("✗")
        hard.append(f"条数不符: {len(data)} != {len(queries)}")

    print(f"[2] 键集与 queries 一致", end="  ")
    if set(data) == set(queries):
        print("OK")
    else:
        print("✗")
        miss = set(queries) - set(data)
        extra = set(data) - set(queries)
        hard.append(f"键集不符（缺 {len(miss)} / 多 {len(extra)}）")

    print("[3] 字段集与回显", end="  ")
    nf = n_echo = 0
    for qid, v in data.items():
        if set(v) != FIELDS:
            nf += 1
            continue
        q0 = queries[qid]
        if any(v[f] != q0[f] for f in ("visible", "infrared", "depth", "query")):
            n_echo += 1
    if nf == 0 and n_echo == 0:
        print("OK（四字段全部原样回显）")
    else:
        print(f"✗ 字段集异常 {nf} / 回显不一致 {n_echo}")
        hard.append(f"字段/回显问题: 字段集异常 {nf}, 回显不一致 {n_echo}")

    print("[4] bbox 合法性", end="  ")
    k = Counter()
    for v in data.values():
        b = v.get("bbox")
        if b is None:
            k["null"] += 1
            continue
        if not (isinstance(b, list) and len(b) == 4):
            k["非四元"] += 1
            continue
        if not all(isinstance(x, (int, float)) and x == x for x in b):
            k["NaN/非数值"] += 1
            continue
        if not ordered(b):
            k["逆序或越界"] += 1
            continue
        k["合法"] += 1
    bad = {x: n for x, n in k.items() if x != "合法" and x != "null"}
    if not bad:
        print(f"OK（合法 {k['合法']}）")
    else:
        print(f"✗ {bad}")
        hard.append(f"bbox 非法: {bad}")

    if k["null"]:
        print(f"[5] bbox 为 null  {k['null']} 条  ← 待决策项，不计本项失败")
        soft.append(f"{k['null']} 条无框")

    print("\n=== 结论 ===")
    if hard:
        for h in hard:
            print("  ✗", h)
        print("  校验不通过")
    else:
        print("  ✓ 全部硬性约束通过")
    for s in soft:
        print("  !", s)
    return 1 if hard else 0


if __name__ == "__main__":
    sys.exit(main())
