#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐条选择协议出图：蓝=q38 件（回复 1），红=annotate 件（回复 2）。

用法
    python _pick_show.py                 # 战场清单前 20 条（qid 升序，从第 1 条起）
    python _pick_show.py --start 20      # 下一批
    python _pick_show.py --n 30 --list _pick_qids.json   # 换作用域清单

输出
    pick_viz/_sheet_batch.png   本批拼版（每张子图标序号/qid/IoU/查询）
    pick_viz/row_<idx>_<qid>.png 单图
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import _viz_q38 as V  # noqa: E402  复用其 _annotate/_contact_sheet/_wrap/_iou

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
BLUE = R / "submission_zips" / "semifinal" / "0.7963" / "result_semi_q38_nonord.json"
RED_ZIP = R / "submission_zips" / "semifinal" / "未提交" / "result_semi_27b_annotate.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="_pick_battle.json")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    q = json.loads((DATA / "queries" / "queries.json").read_text("utf-8"))
    blue = json.loads(BLUE.read_text("utf-8"))
    with zipfile.ZipFile(RED_ZIP) as zf:
        red = json.loads(zf.read("submission.json").decode("utf-8"))
    ks = json.loads((R / a.list).read_text("utf-8"))
    batch = ks[a.start:a.start + a.n]
    if not batch:
        print("清单已到末尾")
        return 1

    odir = R / "pick_viz"
    odir.mkdir(exist_ok=True)
    items = []
    for i, k in enumerate(batch):
        src = DATA / q[k]["visible"]
        if not src.exists():
            print(f"  ✗ 缺图 {src}")
            continue
        b0, c0 = blue[k]["bbox"], red[k]["bbox"]
        io = V._iou(b0, c0)
        lines = [f"[{a.start + i + 1}] {k}  IoU(蓝,红)={io:.2f}  "
                 f"1=蓝 2=红", *[f"Q: {t}" for t in V._wrap(q[k]["query"])]]
        ann = V._annotate(Image.open(src), b0, c0, lines)
        ann.save(odir / f"row_{a.start + i + 1:04d}_{k}.png")
        items.append((f"#{a.start + i + 1} {k} IoU={io:.2f}", ann))
        print(f"  #{a.start + i + 1} {k}  IoU={io:.2f}  {q[k]['query'][:60]}")

    V._contact_sheet(items, str(odir / "_sheet_batch.png"), cols=2)
    print(f"\n本批 {len(items)} 条｜范围 {a.start + 1}..{a.start + len(batch)}"
          f"｜清单共 {len(ks)}")
    print(f"  拼版 {odir / '_sheet_batch.png'}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
