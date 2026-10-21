#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""切出两份分层的"件"，供 `_viz_q38.py` 直接复用（不新写可视化实现）。

· `_m_piece_ord.json` = 基线（0.7963）＋ **只施加 M 在序数层的改动**
· `_m_piece_non.json` = 基线（0.7963）＋ **只施加 M 在非序数层的改动**

这样跑两遍 `_viz_q38.py` 就能分别看"序数 484 条"与"非序数 846 条"各自画面，
从而用**眼睛**判断 M 到底是"序数层被打坏"还是"全局退化"。
"""
from __future__ import annotations

import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal"
M = SUB / "未提交" / "result_semi_m3m_rules5_mb.json"
BEST = SUB / "0.7963" / "result_semi_q38_nonord.json"
CORES = R / "_ordinal_cores_r2.json"


def main() -> int:
    m = json.loads(M.read_text(encoding="utf-8"))
    b = json.loads(BEST.read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))

    chg = [k for k in m if m[k]["bbox"] != b[k]["bbox"]]
    ord_c = [k for k in chg if k in cores]
    non_c = [k for k in chg if k not in cores]
    print(f"改动 {len(chg)} = 序数 {len(ord_c)} + 非序数 {len(non_c)}")

    for name, ks in (("_m_piece_ord.json", set(ord_c)),
                     ("_m_piece_non.json", set(non_c))):
        out = {k: dict(v) for k, v in b.items()}
        for k in ks:
            out[k]["bbox"] = m[k]["bbox"]
        p = R / name
        p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        assert len(out) == 5690
        print(f"  已写 {p.name}（施加 {len(ks)} 条改动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
