#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对最终仍无框的 19 条，用 Qwen3-VL-8B 换输入/换问法再试一轮。

设计（每条按序尝试，取第一个出框者并记录来源）：
  1) **语义匹配的模态图** + 原提示词   —— 热成像类→红外，远近类→深度，无线索→可见光
  2) 其余两张模态图 + 原提示词         —— 替代视角（无线索的 16 条在可见光上已失败）
  3) 可见光 + 第二条提示词（含"给最优猜测"）—— 8B 此前是**显式拒答**
     （"There is no object in the image that matches…"），换个问法打破拒答

判据复用：提示词与框解析一律从 `_probe_models` 导入（那是与线上逐字相同的版本），
不在此重写；数值量化沿用 6 位小数。

用法（解释器用 myconda，与其他 8B 作业一致）：
    /root/miniconda3/envs/myconda/bin/python -u /root/_retry_19_qwen.py
输出：/srv/output/retry19.json
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, "/root")

from PIL import Image  # noqa: E402

from _probe_models import FULL_DIRECT_PROMPT, Hf8B, parse_boxes  # noqa: E402

MODALITIES = ("visible", "infrared", "depth")
TH = re.compile(r"thermal|infrared|heat signature|warm|warmer|warmest|heat|hot|"
                r"cold|temperature|emitting", re.I)
DP = re.compile(r"closest|farthest|furthest|nearest|in front|behind|depth|distance",
                re.I)

ALL_PROMPT = (
    "Find every instance of the following object in this image and output all "
    'their bounding boxes.\nObject: "{q}"\n'
    'Output ONLY a JSON array like [{{"bbox_2d": [x1, y1, x2, y2], "label": '
    '"object"}}] where coordinates are integers normalized to 0-1000 '
    "(x1,y1 = top-left, x2,y2 = bottom-right). "
    "If you are uncertain, give your best guess for the most likely location."
)


def semantic_modality(query: str) -> str:
    """按问句线索给出首选模态；无线索时用可见光（与线上一致）。"""
    if TH.search(query):
        return "infrared"
    if DP.search(query):
        return "depth"
    return "visible"


def load_img(rel: str):
    im = Image.open(os.path.join("/srv/dataset", rel)).convert("RGB")
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default="/srv/output/rest19_qids.json")
    ap.add_argument("--out", default="/srv/output/retry19.json")
    ap.add_argument("--alt-only", action="store_true",
                    help="模态尝试顺序里不含 visible（用于条目已有可见光框的场景）")
    ap.add_argument("--modalities", default=",".join(MODALITIES),
                    help="限定可用的模态，逗号分隔；如 'infrared' 只试红外")
    a = ap.parse_args()
    allowed = tuple(m.strip() for m in a.modalities.split(",") if m.strip())

    queries = json.load(open("/srv/dataset/queries/queries.json",
                            encoding="utf-8"))
    qids = json.load(open(a.qids, encoding="utf-8"))
    print(f"待补 {len(qids)} 条 | 输出 {a.out} | alt_only={a.alt_only}",
          flush=True)

    res = {}
    if os.path.exists(a.out):
        try:
            res = json.load(open(a.out, encoding="utf-8"))
            print(f"已加载既有 {len(res)} 条", flush=True)
        except Exception:
            res = {}
    todo = [k for k in qids if k not in res]

    eng = Hf8B()

    t0 = time.time()
    for i, k in enumerate(todo, 1):
        rec = queries[k]
        q = rec["query"]
        first = semantic_modality(q)
        order = [first] + [m for m in MODALITIES if m != first]
        order = [m for m in order if m in allowed]
        if a.alt_only:
            order = [m for m in order if m != "visible"]
        rec_out = {"query": q, "semantic_modality": first, "tries": []}
        got = None
        for mod in order:
            try:
                img = load_img(rec[mod])
                txt = eng.gen(img, FULL_DIRECT_PROMPT.format(q=q))
                bs = parse_boxes(txt)
                hit = ([round(v, 6) for v in bs[0]] if bs else None)
            except Exception as ex:  # noqa: BLE001
                hit, txt = None, f"<err {type(ex).__name__}>"
            rec_out["tries"].append({"modality": mod, "prompt": "direct",
                                     "hit": hit, "raw": str(txt)[:120]})
            if hit:
                got = ("direct", mod, hit)
                break
        if got is None and not a.alt_only:
            # 换问法：8B 此前是显式拒答，故加"给最优猜测"打破拒答
            try:
                img = load_img(rec["visible"])
                txt = eng.gen(img, ALL_PROMPT.format(q=q))
                bs = parse_boxes(txt)
                hit = ([round(v, 6) for v in bs[0]] if bs else None)
            except Exception as ex:  # noqa: BLE001
                hit, txt = None, f"<err {type(ex).__name__}>"
            rec_out["tries"].append({"modality": "visible", "prompt": "all_guess",
                                     "hit": hit, "raw": str(txt)[:120]})
            if hit:
                got = ("all_guess", "visible", hit)

        if got:
            rec_out.update({"bbox": got[2], "source": f"{got[0]}@{got[1]}"})
        else:
            rec_out.update({"bbox": None, "source": None})
        res[k] = rec_out

        if i % 5 == 0 or i == len(todo):
            json.dump(res, open(a.out, "w", encoding="utf-8"),
                      ensure_ascii=False)
            el = time.time() - t0
            nb = sum(1 for v in res.values() if v.get("bbox"))
            eta = (len(todo) - i) / max(i / max(el, 1e-9), 1e-9) / 60
            print(f"[{len(res)}/{len(qids)}] 本轮 {i}/{len(todo)} "
                  f"出框 {nb} 用时 {el/60:.1f}min ETA {eta:.1f}min", flush=True)

    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    ok = {k: v for k, v in res.items() if v.get("bbox")}
    print(f"\n完成 {len(res)} 条 | 出框 {len(ok)} | 仍无框 {len(res)-len(ok)}",
          flush=True)
    if len(res) <= 25:
        for k, v in res.items():
            print(f"  {k} -> {v.get('source')} {v.get('bbox')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
