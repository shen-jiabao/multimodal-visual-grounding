#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL-8B-Instruct 探针：对 38 条探针问句做 grounding，验证接口/格式/耗时。

输出 /srv/output/qwen3vl_probe.json（含 raw 原文，便于确认输出格式）
"""
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/srv/models/Qwen3-VL-8B-Instruct"
DATA_ROOT = "/srv/dataset_probe"
QUERIES = os.path.join(DATA_ROOT, "mini_queries.json")
OUT = "/srv/output/qwen3vl_probe.json"
MAX_NEW_TOKENS = 256

# 输出格式兼容：JSON bbox_2d / <box> / <|box_start|>(x,y),(x,y)<|box_end|>
RE_BBOX_JSON = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
RE_BOX_TAG = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
RE_QWEN_BOX = re.compile(r"<\|box_start\|>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*,\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*<\|box_end\|>")


def parse_bbox(text: str):
    """从模型输出解析首个 bbox，归一化到 0-1（0-1000 口径）。"""
    for rx in (RE_BBOX_JSON, RE_BOX_TAG, RE_QWEN_BOX):
        m = rx.search(text)
        if m:
            x1, y1, x2, y2 = map(int, m.groups())
            return [x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0]
    return None


def build_prompt(query: str) -> str:
    """grounding 提示词：要求 JSON bbox_2d，坐标 0-1000。"""
    return (
        f"Locate the object described below in this image and output its "
        f"bounding box.\nDescription: {query}\n"
        f"Output ONLY a JSON array like "
        f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}] '
        f"where coordinates are integers normalized to 0-1000 "
        f"(x1,y1 = top-left, x2,y2 = bottom-right)."
    )


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    queries = json.load(open(QUERIES, encoding="utf-8"))
    print(f"[probe] 问句 {len(queries)}", flush=True)

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto")
    model.eval()
    print(f"[probe] 模型加载 {time.time()-t0:.1f}s; "
          f"显存 {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    dev = "cuda"
    img_cache = {}
    results = {}
    times = []
    n_ok = 0
    for i, (k, rec) in enumerate(queries.items(), 1):
        img_rel = rec["visible"]
        if img_rel not in img_cache:
            im = Image.open(os.path.join(DATA_ROOT, img_rel)).convert("RGB")
            img_cache[img_rel] = im
        img = img_cache[img_rel]
        prompt = build_prompt(rec["query"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(dev)
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False)
        dt = time.time() - t1
        times.append(dt)
        trimmed = [o[len(inp):] for inp, o in zip(inputs.input_ids, gen)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        bbox = parse_bbox(text)
        if bbox:
            n_ok += 1
        results[k] = {
            "query": rec["query"][:120],
            "img": img_rel,
            "time_s": round(dt, 3),
            "bbox": [round(v, 4) for v in bbox] if bbox else None,
            "raw": text[:400],
        }
        if i % 10 == 0 or i == len(queries):
            print(f"[probe] {i}/{len(queries)} 命中 {n_ok} "
                  f"最近 {dt:.2f}s", flush=True)

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    avg = sum(times) / len(times) if times else 0
    print("\n===== Qwen3-VL 探针结论 =====")
    print(f"问句 {len(queries)}，命中(有框) {n_ok} "
          f"({100*n_ok/len(queries):.1f}%)")
    print(f"单条耗时 均值 {avg:.2f}s P95 "
          f"{sorted(times)[int(len(times)*0.95)-1]:.2f}s")
    print(f"全量9555 预估 {avg*9555/3600:.1f}h")
    miss = [k for k, v in results.items() if v["bbox"] is None]
    print(f"无框 {len(miss)}: {miss[:5]}")
    for k in miss[:3]:
        print(f"  [miss] {k}: {results[k]['raw'][:150]!r}")
    print(f"输出 {OUT}")


if __name__ == "__main__":
    main()