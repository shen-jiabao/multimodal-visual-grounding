#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装配"人眼选择件"：基体 = annotate（红），选蓝位贴 q38 框。

规则
    · _pick_choices.json 中值为 "blue" 的 qid ⇒ 用 q38 件的 bbox
    · 其余全部（含"red"与未判定）⇒ 用 annotate 件的 bbox
    · 四字段回显（visible/infrared/depth/query）沿用 annotate 件
输出
    submission_zips/semifinal/未提交/result_semi_pick_v1.json / .zip + SHA256
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
BLUE = R / "submission_zips" / "semifinal" / "0.7963" / "result_semi_q38_nonord.json"
RED_ZIP = R / "submission_zips" / "semifinal" / "未提交" / "result_semi_27b_annotate.zip"
CH = R / "_pick_choices.json"
OUT = R / "submission_zips" / "semifinal" / "未提交"


def main() -> int:
    blue = json.loads(BLUE.read_text("utf-8"))
    with zipfile.ZipFile(RED_ZIP) as zf:
        red = json.loads(zf.read("submission.json").decode("utf-8"))
    ch = json.loads(CH.read_text("utf-8"))
    assert set(blue) == set(red), "键集不一致"

    out = {}
    n_blue = 0
    for k, v in red.items():
        if ch.get(k) == "blue":
            v = dict(v)
            v["bbox"] = blue[k]["bbox"]
            n_blue += 1
        out[k] = v

    pj = OUT / "result_semi_pick_v1.json"
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    pz = OUT / "result_semi_pick_v1.zip"
    with zipfile.ZipFile(pz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "submission.json")
    sha = hashlib.sha256(pz.read_bytes()).hexdigest().upper()
    print(f"条目 {len(out)}｜贴蓝 {n_blue}（已判蓝全部生效）｜其余红")
    print(f"zip {pz.name}  SHA256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
