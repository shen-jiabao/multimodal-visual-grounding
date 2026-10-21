#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记录逐条选择：--blue 给全局序号（逗号分隔）=蓝，本批其余=红。

累积写入 _pick_choices.json（qid → "blue"/"red"），装配器据此出件。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
CH = R / "_pick_choices.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="_pick_qids.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--blue", default="", help="选蓝的全局序号，如 11,13")
    a = ap.parse_args()

    ks = json.loads((R / a.list).read_text("utf-8"))
    batch = ks[a.start:a.start + a.n]
    ch = json.loads(CH.read_text("utf-8")) if CH.exists() else {}
    blues = {int(x) for x in a.blue.split(",") if x.strip()}
    for i, k in enumerate(batch):
        ch[k] = "blue" if (a.start + i + 1) in blues else "red"
    CH.write_text(json.dumps(ch, ensure_ascii=False), "utf-8")
    nb = sum(1 for v in ch.values() if v == "blue")
    print(f"记录本批 {len(batch)} 条｜累计已选 {len(ch)}｜蓝 {nb} 红 {len(ch)-nb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
