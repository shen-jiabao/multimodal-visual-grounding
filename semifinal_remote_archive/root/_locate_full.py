#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LocateAnything-3B 全量推理：9555 问句逐条以 query 为 prompt 生成 bbox。

- 复用 px1 已验证接口：sdpa 注意力 / slow 生成 / bfloat16 / 0-1000 归一化
- 每 100 条写一次 checkpoint（可断点续跑），远端进程结束即视为完成
- 输出 /srv/output/locate_bbox_all.json
"""
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig

MODEL_PATH = "/srv/models/LocateAnything-3B"
DATA_ROOT = "/srv/dataset"
QUERIES = os.environ.get("QUERIES_JSON",
                         "/srv/dataset/queries/queries.json")
OUT_JSON = os.environ.get("OUT_JSON", "/srv/output/locate_bbox_all.json")
CHKPT_JSON = OUT_JSON  # checkpoint = 结果文件本身
MAX_IMAGE_SIZE = 2500
MAX_NEW_TOKENS = 256  # 单目标描述定位，够用；密集场景可放宽
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
BATCH_SAVE = 50  # 10.2h 长作业：50 条落盘降低中断损失


def parse_boxes(answer: str):
    """解析模型输出中的全部归一化框（0-1000 → 0-1），保持出现顺序。"""
    out = []
    for m in BOX_RE.finditer(answer):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return out


def norm_bbox(b):
    """归一化 + 保证 x1<x2, y1<y2。"""
    x1, y1, x2, y2 = b
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    return [x1, y1, min(1.0, max(x2, x1 + 1e-3)),
            min(1.0, max(y2, y1 + 1e-3))]


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    print("加载完成; 文本 attn =",
          model.language_model.config._attn_implementation, flush=True)
    return device, tokenizer, processor, model


def infer_qid(device, tokenizer, processor, model, img, query: str):
    """单问句推理：返回(首框, 框总数, 全部归一化框, 原始输出)。

    `all_boxes` 必须落盘：复赛融合采用「互印证」取框规则——多框时需拿全部
    候选去比对照答案的 IoU。早期版本只存 boxes[0]，会让该规则无从执行。
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": query},
    ]}]
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    images, _ = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images,
                       return_tensors="pt").to(device)
    pixel_values = inputs["pixel_values"].to(model.dtype)
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
    if isinstance(resp, tuple):
        answer = resp[0][0] if isinstance(resp[0], list) else resp[0]
    else:
        answer = resp[0] if isinstance(resp, list) else resp
    answer = str(answer)

    # 存全部框（不只第一个）：融合阶段要做去重与互印证选框
    boxes = parse_boxes(answer)
    if boxes:
        all_boxes = [norm_bbox(b) for b in boxes]
        b, n_all = all_boxes[0], len(boxes)
    else:
        all_boxes, b, n_all = [], None, 0
    return b, n_all, all_boxes, answer


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    queries = json.load(open(QUERIES, encoding="utf-8"))
    total = len(queries)
    print(f"问句总数 {total}", flush=True)

    # 加载已有 checkpoint（断点续跑）
    results = {}
    if os.path.exists(CHKPT_JSON):
        try:
            results = json.load(open(CHKPT_JSON, encoding="utf-8"))
            print(f"已加载 checkpoint {len(results)} 条", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"checkpoint 解析失败，从头开始: {ex!r}", flush=True)
            results = {}

    keys = list(queries.keys())
    todo = [k for k in keys if k not in results]
    if not todo:
        print("全部完成，退出", flush=True)
        return
    print(f"待处理 {len(todo)}/{total}", flush=True)

    device, tokenizer, processor, model = load_model()

    # 缓存图像
    img_cache = {}
    done = len(results)

    def get_img(img_rel: str):
        if img_rel not in img_cache:
            p = os.path.join(DATA_ROOT, img_rel)
            im = Image.open(p).convert("RGB")
            w0, h0 = im.size
            if max(w0, h0) > MAX_IMAGE_SIZE:
                s = MAX_IMAGE_SIZE / max(w0, h0)
                im = im.resize((int(w0 * s), int(h0 * s)), Image.LANCZOS)
            img_cache[img_rel] = im
            if len(img_cache) > 32:
                img_cache.clear()  # 防止显存堆积
        return img_cache[img_rel]

    t0 = time.time()
    for i, k in enumerate(todo, 1):
        rec = queries[k]
        try:
            img = get_img(rec["visible"])
            b, n_all, all_boxes, raw = infer_qid(
                device, tokenizer, processor, model, img, rec["query"])
            results[k] = {
                "bbox": b,
                "n_all_boxes": n_all,
                "bbox_all": all_boxes,
                "raw": raw[:200] if raw else "",
            }
        except Exception as ex:  # noqa: BLE001
            print(f"[ERR] {k}: {ex!r}", flush=True)
            results[k] = {"bbox": None, "n_all_boxes": 0, "bbox_all": [],
                          "raw": "", "err": str(ex)}

        done += 1
        if i % BATCH_SAVE == 0 or i == len(todo):
            with open(CHKPT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
            el = time.time() - t0
            spd = i / max(el, 1e-6)
            eta = (len(todo) - i) / max(spd, 1e-6) / 60
            hits = sum(1 for v in results.values() if v.get("bbox"))
            print(f"[{done}/{total}] 本轮 {i}/{len(todo)} "
                  f"{spd:.2f} q/s 命中 {hits} ETA {eta:.1f}min", flush=True)

    with open(CHKPT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    nbbox = sum(1 for v in results.values() if v.get("bbox"))
    print(f"完成 {len(results)} 条, 命中 {nbbox}, 无框 {len(results)-nbbox}", flush=True)


if __name__ == "__main__":
    main()