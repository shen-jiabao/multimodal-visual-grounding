#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红外模态补轮：仅对 8B 无框的条目，用**红外图**重跑 LocateAnything-3B。

动机（实测）：45 条无框里 27% 的问句问的是热成像/红外内容，而原管线只喂了
可见光图 —— 模型在 RGB 图上找不到"热信号"于是拒答。这不是模型弱，是喂错模态。
数据集本身提供 Images/infrared/，初赛的三模态早期融合也用过 IR，故不是新发明。

复用 `_locate_full.py` 的 `load_model` / `infer_qid`，判据不另写一份。
图像缩放上限保持一致（MAX_IMAGE_SIZE=2500），只把输入源换成红外图。

用法：
    /srv/venv-la2/bin/python -u /root/_retry_ir.py
输出：/srv/output/retry_ir.json（含 bbox_all，供融合的互印证规则使用）
"""
import json
import os
import sys
import time

sys.path.insert(0, "/root")

from PIL import Image  # noqa: E402

import _locate_full as L  # noqa: E402

OUT = "/srv/output/retry_ir.json"
NOBOX = "/srv/output/nobox_qids.json"


def main() -> int:
    queries = json.load(open("/srv/dataset/queries/queries.json",
                            encoding="utf-8"))
    if not os.path.exists(NOBOX):
        sys.exit(f"缺少无框清单 {NOBOX}")
    qids = json.load(open(NOBOX, encoding="utf-8"))
    print(f"待补条目 {len(qids)} 条", flush=True)

    # 断点续跑
    res = {}
    if os.path.exists(OUT):
        try:
            res = json.load(open(OUT, encoding="utf-8"))
            print(f"已加载既有结果 {len(res)} 条", flush=True)
        except Exception as ex:
            print(f"既有结果解析失败，从头开始: {ex!r}", flush=True)
    todo = [k for k in qids if k not in res]

    device, tokenizer, processor, model = L.load_model()

    def get_img(rel: str):
        im = Image.open(os.path.join("/srv/dataset", rel)).convert("RGB")
        w0, h0 = im.size
        if max(w0, h0) > L.MAX_IMAGE_SIZE:
            s = L.MAX_IMAGE_SIZE / max(w0, h0)
            im = im.resize((int(w0 * s), int(h0 * s)), Image.LANCZOS)
        return im

    t0 = time.time()
    for i, k in enumerate(todo, 1):
        rec = queries[k]
        entry = {"query": rec["query"], "visible": rec["visible"],
                 "infrared": rec["infrared"], "source": "infrared"}
        try:
            img = get_img(rec["infrared"])
            b, n_all, all_boxes, raw = L.infer_qid(
                device, tokenizer, processor, model, img, rec["query"])
            entry.update({"bbox": b, "n_all_boxes": n_all,
                          "bbox_all": all_boxes, "raw": raw[:200]})
        except Exception as ex:  # noqa: BLE001
            entry.update({"bbox": None, "n_all_boxes": 0, "bbox_all": [],
                          "raw": "", "err": repr(ex)[:200]})
        res[k] = entry

        if i % 5 == 0 or i == len(todo):
            json.dump(res, open(OUT, "w", encoding="utf-8"),
                      ensure_ascii=False)
            el = time.time() - t0
            nb = sum(1 for v in res.values() if v.get("bbox"))
            print(f"[{len(res)}/{len(qids)}] 本轮 {i}/{len(todo)} "
                  f"{i/max(el,1e-9):.2f} q/s 有框 {nb} "
                  f"ETA {(len(todo)-i)/max(i/max(el,1e-9),1e-9)/60:.1f}min",
                  flush=True)

    json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    nb = sum(1 for v in res.values() if v.get("bbox"))
    print(f"\n完成 {len(res)} 条，红外图出框 {nb}，仍无框 {len(res)-nb}",
          flush=True)
    for k, v in res.items():
        if not v.get("bbox"):
            print(f"  [仍无框] {k}: {str(v.get('raw'))[:90]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
