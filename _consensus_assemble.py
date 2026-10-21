#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 路线装配：pick_v1 基体 + 对 annotate 独行位按共识贴 q38 蓝框。

分层策略（用户选保守档）
  · 用户已判（_consensus_calib / _pick_choices）→ 从用户
  · 未判且 vote_q38 >= 30 → 贴 q38（蓝）
  · 其余 → 回 annotate（红）

输出 result_semi_pick_v2.json / .zip + SHA256。
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
SUB = SEMI / "未提交"
VOTE_MIN = 30


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
    base = json.loads((SUB / "result_semi_pick_v1.json").read_text("utf-8"))
    q38 = json.loads((SEMI / "0.7963" / "result_semi_q38_nonord.json")
                     .read_text("utf-8"))
    an = json.loads((SUB / "result_semi_27b_annotate.json").read_text("utf-8"))
    sp = json.loads((R / "_consensus_split.json").read_text("utf-8"))
    calib = json.loads((R / "_consensus_calib.json").read_text("utf-8"))
    ch = json.loads((R / "_pick_choices.json").read_text("utf-8"))

    out = {k: dict(v) for k, v in base.items()}
    n_blue = n_red = 0
    from_user = 0
    for r in sp["annotate_only"]:
        k = r["qid"]
        dec = calib.get(k) or ch.get(k)
        if dec is not None:
            from_user += 1
        elif r["vote_q38"] >= VOTE_MIN:
            dec = "blue"
        else:
            dec = "red"
        if dec == "blue":
            out[k]["bbox"] = list(q38[k]["bbox"])
            n_blue += 1
        else:
            out[k]["bbox"] = list(an[k]["bbox"])
            n_red += 1

    # 校验：格式合法
    for k, e in out.items():
        b = e["bbox"]
        assert len(b) == 4 and all(isinstance(x, (int, float)) for x in b), k
        assert 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1, (k, b)

    diff = sum(1 for k in out if out[k]["bbox"] != base[k]["bbox"])
    same_q38 = sum(1 for k in out if _iou(out[k]["bbox"], q38[k]["bbox"]) >= 0.5)
    pj = SUB / "result_semi_pick_v2.json"
    pj.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    pz = SUB / "result_semi_pick_v2.zip"
    with zipfile.ZipFile(pz, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pj, "submission.json")
    sha = hashlib.sha256(pz.read_bytes()).hexdigest().upper()
    print(f"条目 {len(out)}｜独行位处理：贴蓝 {n_blue}｜回红 {n_red}"
          f"（其中用户亲判 {from_user} 条）")
    print(f"与 pick_v1 相比改动 {diff} 条｜与 q38 同框率 {same_q38/len(out)*100:.1f}%")
    print(f"输出 {pj.name} / {pz.name}")
    print(f"zip SHA256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
