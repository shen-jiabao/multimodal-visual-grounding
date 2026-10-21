#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""枚举能力诊断：测 Qwen3-VL 多种 prompt 的候选召回 vs LocateAnything。

目标：找到能稳定枚举多实例的"枚举器"，供 SoM 步骤①使用。
"""
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/srv/models/Qwen3-VL-8B-Instruct"
IMG = "/srv/dataset/Images/visible/000023.png"
CORE = "stone pier"

PROMPTS = {
    "P1_find_every": f"Find every {CORE} in this image. Output a JSON array with one item like "
                     f'{{"bbox_2d": [x1, y1, x2, y2], "label": "{CORE}"}} for EACH {CORE}, '
                     f"ordered from left to right. Coordinates are integers 0-1000.",
    "P2_count_first": f"How many {CORE}s are in this image? First give the number, then output "
                      f"one JSON item {{\"bbox_2d\": [x1,y1,x2,y2]}} per instance, "
                      f"left to right, integers 0-1000.",
    "P3_detect_all": f"Detect ALL instances of \"{CORE}\" in the image. Output their bounding "
                     f"boxes as a JSON array of {{\"bbox_2d\":[x1,y1,x2,y2]}}, coordinates 0-1000.",
    "P4_all_boxes": f"Output bounding boxes for all {CORE}s visible in the image as a JSON "
                    f"array [{{\"bbox_2d\":[x1,y1,x2,y2]}}], integers 0-1000. Include partial ones.",
    "P5_locate_all": f"Locate all {CORE} in the image and return every bounding box in JSON "
                     f'format: [{{"bbox_2d": [x1, y1, x2, y2]}}]. 0-1000 normalized.',
    "P6_grounding_list": f"Please ground the following objects: {CORE}. Output a JSON list, each "
                         f'{{"bbox_2d": [x1,y1,x2,y2], "label": "{CORE}"}}, 0-1000.',
}

RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


def parse(text):
    out = []
    for m in RE_BBOX.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0])
    return out


def main():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto").eval()
    img = Image.open(IMG).convert("RGB")
    print(f"[enum] 图 000023, core={CORE}\n")
    res = {}
    for name, prompt in PROMPTS.items():
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inp = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to("cuda")
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=512, do_sample=False)
        dt = time.time()-t1
        txt = processor.batch_decode(
            [o[len(i):] for i, o in zip(inp.input_ids, gen)],
            skip_special_tokens=True)[0]
        boxes = parse(txt)
        res[name] = {"n": len(boxes), "boxes": boxes, "raw": txt[:300]}
        print(f"[{name}] {len(boxes)} 框 ({dt:.1f}s)")
        for b in boxes:
            print(f"    {[round(v,3) for v in b]}")
        print(f"    RAW: {txt[:160]!r}\n")
    json.dump(res, open("/srv/output/enum_test.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("输出 /srv/output/enum_test.json")


if __name__ == "__main__":
    main()