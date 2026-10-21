#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C 路线：用 30 套历史产物给 annotate vs q38 的战场位（IoU<0.5）投"独立票"。

仅把不同来源的产物算作独立意见（排除基体自身、及其几何衍生件）。
输出：
  · annotate 独行位（只有 annotate 一个唱，其余近似全站 q38）→ 优先人眼复核
  · q38 独行位（反向，说明当前 annotate 基体站得住）→ 可放心排除
  · 争议位（两边都有支持）→ 需人眼看图
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SEMI = R / "submission_zips" / "semifinal"
SUB = SEMI / "未提交"

# 排除：基体本身 + 其几何衍生件（不是独立意见）
SKIP = {"result_semi_27b_annotate.json", "result_semi_27b_jitter.json",
        "result_semi_27b_shrink.json", "result_semi_pick_v1.json"}


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


def norm(obj):
    out = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        bb = None
        if isinstance(v, dict):
            for key in ("bbox", "box", "pred", "prediction"):
                if key in v and isinstance(v[key], (list, tuple)) and len(v[key]) == 4:
                    bb = list(map(float, v[key]))
                    break
        elif isinstance(v, (list, tuple)) and len(v) == 4 and all(
                isinstance(x, (int, float)) for x in v):
            bb = list(map(float, v))
        if bb is not None:
            out[str(k)] = bb
    return out


def main() -> int:
    an = norm(json.loads((SUB / "result_semi_27b_annotate.json").read_text("utf-8")))
    q38 = norm(json.loads((SEMI / "0.7963" / "result_semi_q38_nonord.json")
                          .read_text("utf-8")))
    battle = json.loads((R / "_pick_battle.json").read_text("utf-8"))
    if isinstance(battle, dict):
        battle = list(battle.keys())
    battle = [str(k) for k in battle]

    # 载入独立意见
    files = sorted(glob.glob(str(SEMI / "**" / "result_*.json"), recursive=True))
    files = [p for p in files if "复赛提交材料" not in p
             and os.path.basename(p) not in SKIP]
    voters = {}
    for p in files:
        n = os.path.basename(p)
        if n in voters:          # 同名只取一份
            continue
        try:
            d = norm(json.loads(Path(p).read_text("utf-8")))
        except Exception:
            continue
        if len(d) >= 5000:
            voters[n] = d
    print(f"独立意见源 {len(voters)} 套（≥5000 条）")

    ann_only, q38_only, disputed, tie = [], [], [], []
    for k in battle:
        if k not in an or k not in q38:
            continue
        a, q = an[k], q38[k]
        sa = sq = 0
        for d in voters.values():
            if k not in d:
                continue
            ia, iq = _iou(d[k], a), _iou(d[k], q)
            if ia > iq:
                sa += 1
            elif iq > ia:
                sq += 1
        rec = {"qid": k, "annotate": a, "q38": q,
               "vote_annotate": sa, "vote_q38": sq}
        if sq >= 5 and sa == 0:
            ann_only.append(rec)
        elif sa >= 5 and sq == 0:
            q38_only.append(rec)
        elif sa >= 3 and sq >= 3:
            disputed.append(rec)
        else:
            tie.append(rec)

    print(f"战场位 {len(battle)}｜annotate独行 {len(ann_only)}｜"
          f"q38独行 {len(q38_only)}｜双方争议 {len(disputed)}｜弱票 {len(tie)}")
    out = {"annotate_only": ann_only, "q38_only": q38_only,
           "disputed": disputed, "weak": tie}
    p = R / "_consensus_split.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), "utf-8")
    print(f"已写 {p}")
    print("\nannotate 独行位前 12（优先人眼复核）：")
    for r in sorted(ann_only, key=lambda r: -r["vote_q38"])[:12]:
        print(f"  {r['qid']}  q38票{r['vote_q38']:>2} an票{r['vote_annotate']}  "
              f"q38={[round(x,3) for x in r['q38']]} an={[round(x,3) for x in r['annotate']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
