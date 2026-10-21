#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocateAnything-3B mini 探针：验证加载 / 推理 / 解析 / 耗时。

路径可经环境变量覆盖（缺省即初赛 px2 的路径），便于同一份脚本复用于复赛：
    PROBE_DATA / PROBE_QUERIES / PROBE_OUT
"""
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig

MODEL_PATH = "/srv/models/LocateAnything-3B"
DATA_ROOT = os.environ.get("PROBE_DATA", "/srv/dataset_probe")
QUERIES = os.environ.get("PROBE_QUERIES",
                         os.path.join(DATA_ROOT, "mini_queries.json"))
OUT = os.environ.get("PROBE_OUT", "/srv/output/probe_result.json")
MAX_IMAGE_SIZE = 2500
MAX_NEW_TOKENS = 256
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer: str):
    out = []
    for m in BOX_RE.finditer(answer):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    queries = json.load(open(QUERIES, encoding="utf-8"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe] device={device} 模型加载中...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    if getattr(config, "vision_config", None) is not None:
        config.vision_config._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    print(f"[probe] 模型加载完成 {time.time()-t0:.1f}s; "
          f"文本attn={model.language_model.config._attn_implementation}",
          flush=True)
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"[probe] 加载后显存占用 {vram:.2f} GB", flush=True)

    img_cache = {}
    results = {}
    times = []
    n_ok = 0
    for k, rec in queries.items():
        img_rel = rec["visible"]
        if img_rel not in img_cache:
            p = os.path.join(DATA_ROOT, img_rel)
            im = Image.open(p).convert("RGB")
            w0, h0 = im.size
            if max(w0, h0) > MAX_IMAGE_SIZE:
                s = MAX_IMAGE_SIZE / max(w0, h0)
                im = im.resize((int(w0 * s), int(h0 * s)), Image.LANCZOS)
            img_cache[img_rel] = im
        img = img_cache[img_rel]
        prompt = rec["query"]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, _ = processor.process_vision_info(messages)
        inputs = processor(text=[text], images=images,
                           return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"].to(model.dtype)
        t1 = time.time()
        with torch.no_grad():
            resp = model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True,
                generation_mode="slow",
                do_sample=False,
                verbose=False,
            )
        dt = time.time() - t1
        times.append(dt)
        if isinstance(resp, tuple):
            answer = resp[0][0] if isinstance(resp[0], list) else resp[0]
        else:
            answer = resp[0] if isinstance(resp, list) else resp
        answer = str(answer)
        boxes = parse_boxes(answer)
        if boxes:
            n_ok += 1
        results[k] = {
            "query": prompt[:120],
            "img": img_rel,
            "time_s": round(dt, 3),
            "n_boxes": len(boxes),
            "boxes": [[round(v, 4) for v in b] for b in boxes[:6]],
            "raw": answer[:300],
        }
        if len(results) % 10 == 0:
            print(f"[probe] {len(results)}/{len(queries)} 命中{n_ok} "
                  f"最近耗时{dt:.2f}s", flush=True)

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    avg = sum(times) / len(times) if times else 0
    print("\n===== 探针结论 =====")
    print(f"问句 {len(queries)}，命中(有框) {n_ok} ({100*n_ok/len(queries):.1f}%)")
    print(f"单条平均耗时 {avg:.2f}s；95分位 "
          f"{sorted(times)[int(len(times)*0.95)-1]:.2f}s")
    print(f"全量9555 预估 {avg*9555/3600:.1f}h (按均值)")
    print(f"输出 {OUT}")
    # 失败样例
    miss = [k for k, v in results.items() if v["n_boxes"] == 0]
    print(f"无框 {len(miss)} 条: {miss[:5]}")
    for k in miss[:2]:
        print(f"  [miss] {k}: {results[k]['raw'][:100]!r}")


if __name__ == "__main__":
    main()