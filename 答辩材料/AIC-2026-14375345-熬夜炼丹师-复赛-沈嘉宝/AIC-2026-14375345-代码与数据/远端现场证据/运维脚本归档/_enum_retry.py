#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""序数「枚举召回」重跑：采集多提示下的原始候选框，供本地合并与判据复用。

背景
----
复赛 1103 条序数中，有 258 条枚举失败（`n_cands` 为 0 或 1，`bbox_all` 未落盘）。
其中 **228 条**四个前提里的①②③④都齐备（轴可判、k 可抽、非参照物），
只是候选数 < k —— 即**纯枚举召回不足**，而非问句不可解析。
故重跑只针对这批组合，外加 **120 条控制组**（现有候选 ≥2 的条目）。

控制组的作用（关键）
--------------------
无真值可用时，**唯一能校验"重跑枚举是否可信"的离线手段**是：拿原枚举已有 ≥2
候选的条目当参照，用完全相同的采集流程重跑一遍，再比较"文本推算所选框"是否
一致。保真度高才可把这套采集用于那 228 条。

设计要点
--------
- **远端只采集原始框，绝不做去重/排序/取舍**：合并策略留到本地，改策略无需再占 GPU。
- 断点续跑：每对 (图, core) 落盘一次，二次进入只补缺失的。
- 三种全文提示 + 2×2 重叠分块提示各一次，覆盖不同失败模式。

用法（远端）
    QWEN_ENUM_MODEL=/srv/models/Qwen3-VL-8B-Instruct \
    python -u _enum_retry.py
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = os.environ.get("QWEN_ENUM_MODEL", "/srv/models/Qwen3-VL-8B-Instruct")
DATA = Path(os.environ.get("QWEN_ENUM_DATA", "/srv/dataset"))
OUT = Path(os.environ.get("QWEN_ENUM_OUT", "/srv/output/_enum_raw.json"))
TARGETS = Path(os.environ.get("QWEN_ENUM_TARGETS", "/srv/output/_enum_targets.json"))

SAVE_EVERY = 1          # 每对都落盘：断点粒度 = 一对 ≈ 7 次生成
MAX_NEW_FULL = 640      # 全文提示的 token 上限
MAX_NEW_TILE = 384      # 分块提示的 token 上限（块内目标更少）

# 与初赛/复赛管线完全一致的原始枚举提示（P1）——控制组要拿它当可比基线
P1 = ('Detect ALL instances of "{core}" in the image. Output their bounding '
      'boxes as a JSON array of {{"bbox_2d":[x1,y1,x2,y2]}}, coordinates are '
      'integers 0-1000, ordered from left to right. Include partially visible ones.')
# 先问数量再逐一出框：对"只给一个就收工"的退化行为有抑制作用
P2 = ('How many "{core}" are visible in this image? Respond with "COUNT: n" as '
      'the first line, then output a JSON array of exactly n objects like '
      '{{"bbox_2d":[x1,y1,x2,y2]}} (integers 0-1000) covering every one of '
      'them, ordered from left to right. Include small, distant, occluded or '
      'partially visible ones.')
# 明确要求"不要只给第一个"，并把边界/遮挡目标纳入
P3 = ('List EVERY "{core}" in this image, including small, distant, occluded, '
      'partially visible ones and ones touching the image border. Do not stop '
      'after the first one. Output a JSON array like '
      '[{{"bbox_2d":[x1,y1,x2,y2]}}], coordinates are integers 0-1000, '
      'ordered from left to right.')

# 2×2 分块，边长重叠 15%：小目标与被遮挡目标在整图提示下最易漏
TILES = ((0.000, 0.000, 0.575, 0.575), (0.425, 0.000, 1.000, 0.575),
         (0.000, 0.425, 0.575, 1.000), (0.425, 0.425, 1.000, 1.000))

RE_BBOX = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
# 备用形态：裸四元数组（仅当上面一条毫无命中时才启用，避免重复计数）
RE_BBOX_ALT = re.compile(
    r"\[\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*\]")


def parse_boxes(text: str) -> list:
    """从模型输出里抽候选框，归一化到 [0,1]。"""
    ms = RE_BBOX.findall(text)
    if not ms:
        ms = RE_BBOX_ALT.findall(text)
    out = []
    for x1, y1, x2, y2 in ms:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return out


def crop_tile(img: Image.Image, t: tuple) -> Image.Image:
    """按归一化窗口裁图。"""
    W, H = img.size
    x0, y0, x1, y1 = t
    return img.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))


def lift(boxes: list, t: tuple) -> list:
    """把块内归一化框映射回整图归一化坐标。"""
    x0, y0, x1, y1 = t
    dw, dh = x1 - x0, y1 - y0
    return [[x0 + b[0] * dw, y0 + b[1] * dh, x0 + b[2] * dw, y0 + b[3] * dh]
            for b in boxes]


def main() -> None:
    q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
    tg = json.loads(TARGETS.read_text(encoding="utf-8"))
    qids = {**tg["target"], **tg["control"]}
    pairs = sorted({(q[k]["visible"], tg["target"].get(k) or tg["control"][k])
                    for k in qids})
    pairs = [(p, c.lower()) for p, c in pairs]
    print(f"目标 {len(tg['target'])} 条 | 控制 {len(tg['control'])} 条 | "
          f"待枚举组合 {len(pairs)}", flush=True)

    raw = {}
    if OUT.exists():
        try:
            raw = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    print(f"已有 {len(raw)} 对（断点续跑）", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto").eval()
    print(f"[enum] 模型就绪 显存 {torch.cuda.memory_allocated() / 1e9:.1f}GB",
          flush=True)

    def gen(img: Image.Image, prompt: str, max_new: int) -> str:
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inp = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        return processor.batch_decode(
            [o[len(i):] for i, o in zip(inp.input_ids, g)],
            skip_special_tokens=True)[0]

    todo = [p for p in pairs if f"{p[0]}||{p[1]}" not in raw]
    print(f"待处理 {len(todo)} 对", flush=True)
    n_img_cache = None
    t0 = time.time()
    for i, (rel, core) in enumerate(todo, 1):
        key = f"{rel}||{core}"
        path = DATA / rel
        rec = {"rel": rel, "core": core}
        try:
            img = Image.open(path).convert("RGB")
        except Exception as ex:
            rec["err"] = f"img_open:{ex!r}"[:120]
            raw[key] = rec
            json.dump(raw, OUT.open("w", encoding="utf-8"), ensure_ascii=False)
            print(f"[{i}/{len(todo)}] ERR {key}: {ex!r}", flush=True)
            continue
        rec["size"] = list(img.size)
        for name, pr, mx in (("p1", P1, MAX_NEW_FULL), ("p2", P2, MAX_NEW_FULL),
                             ("p3", P3, MAX_NEW_FULL)):
            try:
                rec[name] = parse_boxes(gen(img, pr.format(core=core), mx))
            except Exception as ex:
                rec[name] = []
                rec[name + "_err"] = str(ex)[:100]
        for j, t in enumerate(TILES, 1):
            try:
                sub = crop_tile(img, t)
                rec[f"t{j}"] = lift(parse_boxes(
                    gen(sub, P3.format(core=core), MAX_NEW_TILE)), t)
            except Exception as ex:
                rec[f"t{j}"] = []
                rec[f"t{j}_err"] = str(ex)[:100]
        raw[key] = rec
        if i % SAVE_EVERY == 0 or i == len(todo):
            json.dump(raw, OUT.open("w", encoding="utf-8"), ensure_ascii=False)
        el = time.time() - t0
        cnt = {k: len(rec.get(k) or []) for k in
               ("p1", "p2", "p3", "t1", "t2", "t3", "t4")}
        print(f"[{i}/{len(todo)}] {core!r:34s} {cnt} "
              f"已用{el / 60:.1f}min ETA{el / i * (len(todo) - i) / 60:.1f}min",
              flush=True)

    json.dump(raw, OUT.open("w", encoding="utf-8"), ensure_ascii=False)
    n_any = sum(1 for v in raw.values()
                if max(len(v.get(k) or []) for k in
                       ("p1", "p2", "p3", "t1", "t2", "t3", "t4")) >= 2)
    print(f"\n完成：{len(raw)} 对，其中至少一种提示给出 ≥2 框的有 {n_any} 对",
          flush=True)
    print(f"输出 {OUT}", flush=True)


if __name__ == "__main__":
    main()
