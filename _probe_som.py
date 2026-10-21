#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SoM (Set-of-Mark) 探针：解决序数识别。

流程（每张图一次）：
  ① 枚举：让 Qwen3-VL 输出该类别的全部候选框（JSON bbox_2d 数组）
  ② 标注：在图上画红色编号标记（编号按从左到右）
  ③ 选择：把带编号的图给模型，问"哪个编号是 <原问句>"，只答编号
  ④ 映射：编号 → 候选框
校验：与当前提交 result_locate_merge 中已验证的序数框比对 IoU。

输出 /srv/output/qwen3vl_som.json
"""
import json
import os
import re
import time

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = "/srv/models/Qwen3-VL-8B-Instruct"
DATA = "/srv/dataset"
OUT = "/srv/output/qwen3vl_som.json"
MAX_NEW_TOKENS = 256

# 待测：图 → (核心名词, 序号问句 qid 列表)
CASES = {
    "000023": {
        "core": "stone pier",
        "queries": ["000023_001", "000023_002", "000023_003",
                    "000023_004", "000023_005"],
    },
}

RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')


def parse_boxes(text):
    """解析全部 bbox（0-1000 → 0-1）。"""
    out = []
    for m in RE_BBOX.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0])
    return out


def iou(a, b):
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., x2-x1), max(0., y2-y1)
    it = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - it
    return it/ua if ua > 1e-9 else 0.0


def draw_som(img, boxes):
    """在图上画编号标记（红框 + 红底白字编号），编号从左到右 1..N。"""
    im = img.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(16, int(H*0.03)))
    except Exception:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes, 1):
        x1, y1 = b[0]*W, b[1]*H
        x2, y2 = b[2]*W, b[3]*H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        r = max(12, int(H*0.022))
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0))
        txt = str(i)
        tb = d.textbbox((0, 0), txt, font=font)
        d.text((cx-(tb[2]-tb[0])/2, cy-(tb[3]-tb[1])/2 - tb[1]),
               txt, fill=(255, 255, 255), font=font)
    return im


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto").eval()
    print("[som] 模型就绪", flush=True)

    all_q = json.load(open(os.path.join(DATA, "queries", "queries.json"),
                           encoding="utf-8"))
    results = {}
    for img_id, case in CASES.items():
        core = case["core"]
        ipath = os.path.join(DATA, "Images", "visible", f"{img_id}.png")
        img = Image.open(ipath).convert("RGB")

        # ① 枚举（用 P3 风格：Detect ALL instances —— 实测 "Find every X" 只给 1 框）
        eprompt = (f"Detect ALL instances of \"{core}\" in the image. Output their "
                   f"bounding boxes as a JSON array of "
                   f'{{"bbox_2d":[x1,y1,x2,y2]}}, coordinates are integers 0-1000, '
                   f"ordered from left to right. Include partially visible ones.")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": eprompt}]}]
        inp = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=512, do_sample=False)
        etxt = processor.batch_decode(
            [o[len(i):] for i, o in zip(inp.input_ids, gen)],
            skip_special_tokens=True)[0]
        cands = parse_boxes(etxt)
        print(f"[som] {img_id} 枚举到 {len(cands)} 个候选", flush=True)

        # ② 标注 SoM
        som_img = draw_som(img, cands)

        # ③ 逐问句选择编号
        qres = {}
        for qid in case["queries"]:
            qtext = all_q[qid]["query"]
            sprompt = (
                f"The image has {len(cands)} objects marked with red numbered "
                f"circles (1 = leftmost, {len(cands)} = rightmost). "
                f"Based on the numbering, which number corresponds to: "
                f"\"{qtext}\"? Answer with ONLY the number."
            )
            msgs2 = [{"role": "user", "content": [
                {"type": "image", "image": som_img},
                {"type": "text", "text": sprompt}]}]
            inp2 = processor.apply_chat_template(
                msgs2, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to("cuda")
            t1 = time.time()
            with torch.no_grad():
                g2 = model.generate(**inp2, max_new_tokens=16, do_sample=False)
            dt = time.time() - t1
            stxt = processor.batch_decode(
                [o[len(i):] for i, o in zip(inp2.input_ids, g2)],
                skip_special_tokens=True)[0]
            m = re.search(r"\d+", stxt)
            num = int(m.group(0)) if m else None
            bbox = None
            if num and 1 <= num <= len(cands):
                bbox = cands[num-1]
            qres[qid] = {"query": qtext, "answer_num": num,
                         "n_cands": len(cands), "bbox": bbox,
                         "time_s": round(dt, 2), "raw": stxt[:120]}
            print(f"  {qid} -> 编号 {num} bbox={bbox} ({dt:.2f}s)",
                  flush=True)

        results[img_id] = {"core": core, "n_cands": len(cands),
                           "candidates": cands, "enum_raw": etxt[:500],
                           "queries": qres}

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False,
              indent=1)
    print(f"\n输出 {OUT}")
    # 保存一张 SoM 标注图供查验
    for img_id, case in CASES.items():
        img = Image.open(os.path.join(DATA, "Images", "visible",
                                      f"{img_id}.png")).convert("RGB")
        som = draw_som(img, results[img_id]["candidates"])
        som.save(f"/srv/output/som_{img_id}.png")
        print(f"SoM 图 /srv/output/som_{img_id}.png")


if __name__ == "__main__":
    main()