#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 收紧版：只对「纯 32 票独行位」+「用户亲判蓝位」贴 q38。

依据（人眼校准）：贴蓝价值在 100% 票与 <100% 票之间存在断崖——
  32:0 段选蓝 82.5%，31:1 段 40%，29:3 段 30%，16:16 段 20%。
故 v2 的 >=30 票档过宽（含 135 条未验证的 31/30 票位，那里贴蓝是负期望）。

判定优先级：用户亲判(calib/choices) > 纯32票段默认贴蓝 > 其余回 annotate。
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
SUB = SEMI / "未提交"
PURE = 32


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


def main() -> int:
    # 记录 31 票段批（全局 60..79）
    ao = json.loads((R / "_consensus_annonly.json").read_text("utf-8"))
    red31 = {4, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 19}
    p = R / "_consensus_calib.json"
    cal = json.loads(p.read_text("utf-8"))
    for i in range(1, 21):
        cal[ao[58 + i]] = "red" if i in red31 else "blue"
    p.write_text(json.dumps(cal, ensure_ascii=False, indent=1), "utf-8")

    base = json.loads((SUB / "result_semi_pick_v1.json").read_text("utf-8"))
    q38 = json.loads((SEMI / "0.7963" / "result_semi_q38_nonord.json")
                     .read_text("utf-8"))
    an = json.loads((SUB / "result_semi_27b_annotate.json").read_text("utf-8"))
    sp = json.loads((R / "_consensus_split.json").read_text("utf-8"))
    ch = json.loads((R / "_pick_choices.json").read_text("utf-8"))

    out = {k: dict(v) for k, v in base.items()}
    n_blue = n_red = n_user = 0
    for r in sp["annotate_only"]:
        k = r["qid"]
        dec = cal.get(k) or ch.get(k)
        if dec is not None:
            n_user += 1
        elif r["vote_q38"] == PURE:
            dec = "blue"
        else:
            dec = "red"
        if dec == "blue":
            out[k]["bbox"] = list(q38[k]["bbox"])
            n_blue += 1
        else:
            out[k]["bbox"] = list(an[k]["bbox"])
            n_red += 1

    for k, e in out.items():
        b = e["bbox"]
        assert len(b) == 4 and all(isinstance(x, (int, float)) for x in b), k
        assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1, (k, b)

    diff = sum(1 for k in out if out[k]["bbox"] != base[k]["bbox"])
    pj = SUB / "result_semi_pick_v3.json"
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    pz = SUB / "result_semi_pick_v3.zip"
    with zipfile.ZipFile(pz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "submission.json")
    sha = hashlib.sha256(pz.read_bytes()).hexdigest().upper()
    print(f"校准累计 {len(cal)} 条｜独行位处理：贴蓝 {n_blue}｜回红 {n_red}"
          f"（用户亲判 {n_user} 条）")
    print(f"与 pick_v1 相比改动 {diff} 条")
    print(f"输出 {pj.name} / {pz.name}")
    print(f"zip SHA256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
