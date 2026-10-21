#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3.8-27B 闸门 G2/G3 冒烟（预登记 I §4）。

判据（**跑前已定死**，见 `_prereg_I.md` §4 与「★G3 口径重述」）：

    G2 可解析 ≥ 16/20（机检：parse_boxes 非空）
    G3 方向正确 ≥ 12/20 通过｜8~11 放行但登记「接地能力弱」｜< 8 停

G3 是**人眼判据**（框是否落在查询描述的目标附近），本脚本只负责把证据摆出来：
    · 逐条打印 qid / 框 / 查询原文
    · 在 visible 图上画出预测框，存单图 + 一张 4×5 拼版（便于一次看完 20 条）

与全量同源：直接复用 `_qwen_full_pipeline` 的 `make_gen_api` / `FULL_DIRECT_PROMPT`
/ `parse_boxes` / `draw_som`，**不另写一份实现**（本项目纪律：判据只实现一处）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load_qids(path: str) -> list[str]:
    """读 qid 列表；兼容 list / dict 两种写法。"""
    d = json.load(open(path, encoding="utf-8"))
    return list(d.keys()) if isinstance(d, dict) else list(d)


def _contact_sheet(items: list[tuple[str, Image.Image]],
                   out_path: str, cols: int = 4) -> None:
    """把 (qid, 图) 拼成一张网格图，左上角标 qid（纯 ASCII，免字体问题）。"""
    if not items:
        return
    tw = 320
    th = max(1, int(items[0][1].height * tw / items[0][1].width))
    pad = 22
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * (th + pad)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    for i, (qid, im) in enumerate(items):
        r, c = divmod(i, cols)
        y = r * (th + pad)
        sheet.paste(im.resize((tw, th)), (c * tw, y + pad))
        d.text((c * tw + 4, y + 5), qid, fill=(255, 255, 255))
    sheet.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="/root/_probe_qids.json")
    ap.add_argument("--data", default=os.environ.get(
        "QWEN_DATA", "/root/rivermind-data/dataset"))
    ap.add_argument("--out", default="/root/rivermind-data/_q38_gate20.json")
    ap.add_argument("--vis-dir", default="/root/rivermind-data/q38_gate_vis")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    import _qwen_full_pipeline as P

    qids = _load_qids(a.qids)[:a.n]
    q = json.load(open(os.path.join(a.data, "queries", "queries.json"),
                       encoding="utf-8"))
    os.makedirs(a.vis_dir, exist_ok=True)

    print(f"引擎 {P.ENGINE} → {P.API_BASE} model={P.API_MODEL}", flush=True)
    print(f"模态={P.MODALITIES}｜规则5={P.RULES5}｜multi_box={P.MULTIBOX}"
          f"(≤{P.MAXBOX})｜max_new={P.MAXNEW}｜thinking={P.THINKING}", flush=True)
    print(f"数据 {a.data}｜条目 {len(qids)}｜出图 {a.vis_dir}", flush=True)

    gen = P.make_gen_api()

    def _imgs(k):
        """按 MODALITIES 组装输入（1=单图／3=分图三张／3m=三模态合并成一张）。

        返回 (可见光底图, 送模型输入)。给 G3 画框用的一律是可见光那张。
        """
        vis = Image.open(os.path.join(a.data, q[k]["visible"])).convert("RGB")
        if P.MODALITIES == "3":
            return vis, [vis,
                         P.load_ir(os.path.join(a.data, q[k]["infrared"])),
                         P.load_depth(os.path.join(a.data, q[k]["depth"]))]
        if P.MODALITIES == "3m":
            return vis, P.merge3(
                vis,
                P.load_ir(os.path.join(a.data, q[k]["infrared"])),
                P.load_depth(os.path.join(a.data, q[k]["depth"])))
        return vis, vis

    res, sheet = {}, []
    t0 = time.time()
    # ★与全量同源：multi_box 开就走 _direct_multi（同图合并前向），否则逐条 _direct_one。
    #   不在此另写一份前向/解析实现（本项目纪律：判据只实现一处）。
    if P.MULTIBOX:
        byimg = defaultdict(list)
        for k in qids:
            byimg[os.path.basename(q[k]["visible"])].append(k)
        batches = [ks[i:i + P.MAXBOX] for ks in byimg.values()
                   for i in range(0, len(ks), P.MAXBOX)]
    else:
        batches = [[k] for k in qids]

    n_done = 0
    for ks in batches:
        vis, img_in = _imgs(ks[0])
        try:
            if P.MULTIBOX:
                out = P._direct_multi(gen, img_in, [q[k]["query"] for k in ks])
            else:
                out = [P._direct_one(gen, img_in, q[ks[0]]["query"])]
        except Exception as ex:
            out = [{"bbox": None, "err": str(ex)[:120]} for _ in ks]
        for k, r in zip(ks, out):
            boxes = [r["bbox"]] if r.get("bbox") else []
            res[k] = {"bbox": boxes[0] if boxes else None,
                      "n_box": len(boxes), "raw": r.get("raw", "")}
            im = vis.copy()
            if boxes:
                im = P.draw_som(vis, boxes)
                sheet.append((k, im))
            im.save(os.path.join(a.vis_dir, f"{k}.png"))
            n_done += 1
            print(f"  {n_done:>2}/{len(qids)} {k} box={len(boxes)}"
                  f" {[round(v, 3) for v in boxes[0]] if boxes else '—'}",
                  flush=True)
    print(f"  用时 {time.time()-t0:.1f} s", flush=True)

    json.dump(res, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    if sheet:
        _contact_sheet(sheet, os.path.join(a.vis_dir, "_sheet.png"))

    ok = [k for k, v in res.items() if v["bbox"]]
    # ★ 越界检查：G2 的字面判据只要求"可解析"，但**框非法（坐标越界）照样能解析**。
    #   K 件实测就踩过这条（Qwen2.5-VL 输出 smart-resize 像素坐标）⇒ 必须单独报。
    oob = [k for k, v in res.items()
           if v["bbox"] and not all(0.0 <= x <= 1.0 for x in v["bbox"])]
    print("\n=== G2（机检）===")
    print(f"  可解析（bbox 非空）：{len(ok)}/{len(qids)}"
          f"（要求 ≥16）→ {'通过' if len(ok) >= 16 else '★不通过'}")
    print(f"  坐标越界（任一值超出 [0,1]）：{len(oob)}/{len(qids)}"
          f" → {'★不通过（输出约定不符，需 QWEN_BBOX_SPACE 适配）' if oob else '通过'}")

    print("\n=== G3（人眼：框是否落在查询描述的目标附近）===")
    for k in qids:
        b = res[k]["bbox"]
        bs = ("[%.3f %.3f %.3f %.3f]" % tuple(b)) if b else "—"
        print(f"  {k}  {bs}")
        print(f"       Q: {q[k]['query']}")
    print(f"\n  拼版图 {os.path.join(a.vis_dir, '_sheet.png')}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
