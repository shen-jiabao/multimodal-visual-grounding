#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-VL 全量管线：① 全部问句直接推理 ② 序数问句走 SoM 三段法。

- 按图分组处理（每图只加载一次，复用图像）
- Phase A：每条问句 direct grounding → bbox
- Phase B：序数问句 → 按 (图,核心词) 枚举候选 → 打编号 → 选编号
- 双 checkpoint（qwen3vl_all.json / qwen3vl_som_all.json），可断点续跑

路径可经环境变量覆盖（缺省即线上值），用途是**冒烟测试**——可指向小型
临时数据集与临时输出目录，在不污染正式产物的前提下验证整条接线。
    QWEN_MODEL / QWEN_DATA / QWEN_OUT_DIRECT / QWEN_OUT_SOM / QWEN_CORES
"""
import json
import os
import re
import time
from collections import defaultdict

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_PATH = os.environ.get("QWEN_MODEL", "/srv/models/Qwen3-VL-8B-Instruct")
DATA = os.environ.get("QWEN_DATA", "/srv/dataset")
OUT_DIRECT = os.environ.get("QWEN_OUT_DIRECT", "/srv/output/qwen3vl_all.json")
OUT_SOM = os.environ.get("QWEN_OUT_SOM", "/srv/output/qwen3vl_som_all.json")
CORES = os.environ.get("QWEN_CORES", "/srv/output/_ordinal_cores_v3.json")
SAVE_EVERY = 50

FULL_DIRECT_PROMPT = (
    "Locate the object described below in this image and output its bounding "
    "box.\nDescription: {q}\n"
    "Output ONLY a JSON array like "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}] '
    "where coordinates are integers normalized to 0-1000 "
    "(x1,y1 = top-left, x2,y2 = bottom-right)."
)
ENUM_PROMPT = (
    'Detect ALL instances of "{core}" in the image. Output their bounding '
    'boxes as a JSON array of {{"bbox_2d":[x1,y1,x2,y2]}}, coordinates are '
    "integers 0-1000, ordered from left to right. Include partially visible ones."
)
SEL_PROMPT = (
    "The image has {n} objects marked with red numbered circles "
    "(1 = leftmost, {n} = rightmost). Based on the numbering, which number "
    'corresponds to: "{q}"? Answer with ONLY the number.'
)

RE_BBOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
RE_NUM = re.compile(r"\d+")


def parse_boxes(text):
    out = []
    for m in RE_BBOX.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0])
    return out


def draw_som(img, boxes):
    im = img.copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(16, int(H * 0.03)))
    except Exception:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes, 1):
        x1, y1, x2, y2 = b[0]*W, b[1]*H, b[2]*W, b[3]*H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        r = max(12, int(H*0.022))
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0))
        tb = d.textbbox((0, 0), str(i), font=font)
        d.text((cx-(tb[2]-tb[0])/2, cy-(tb[3]-tb[1])/2 - tb[1]),
               str(i), fill=(255, 255, 255), font=font)
    return im


def main():
    os.makedirs(os.path.dirname(OUT_DIRECT), exist_ok=True)
    q = json.load(open(os.path.join(DATA, "queries", "queries.json"),
                       encoding="utf-8"))
    cores = json.load(open(CORES, encoding="utf-8")) if os.path.exists(CORES) else {}
    ordinal = set(cores)

    direct = {}
    if os.path.exists(OUT_DIRECT):
        try:
            direct = json.load(open(OUT_DIRECT, encoding="utf-8"))
        except Exception:
            direct = {}
    som = {}
    if os.path.exists(OUT_SOM):
        try:
            som = json.load(open(OUT_SOM, encoding="utf-8"))
        except Exception:
            som = {}

    # 按图分组
    by_img = defaultdict(list)
    for k, v in q.items():
        by_img[os.path.basename(v["visible"])].append(k)
    imgs = sorted(by_img)
    print(f"图 {len(imgs)}，问句 {len(q)}，序数 {len(ordinal)}", flush=True)
    print(f"已完成 direct={len(direct)} som={len(som)}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, dtype="auto", device_map="auto").eval()
    print(f"[pipe] 模型就绪 显存 {torch.cuda.memory_allocated()/1e9:.1f}GB",
          flush=True)

    def gen(img, prompt, max_new=256):
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

    # 断点：direct 未齐 或 序数 SoM 未齐 的图都要处理。
    # 注意：两个阶段内部各自幂等（direct 命中即跳过、som 命中即跳过），
    # 因此二次进入同一张图只会补齐缺失的那部分，不会重算。
    def _pending(im):
        if any(k not in direct for k in by_img[im]):
            return True
        return any(k in ordinal and k not in som for k in by_img[im])

    todo_imgs = [im for im in imgs if _pending(im)]
    print(f"待处理图 {len(todo_imgs)}", flush=True)

    t0 = time.time()
    n_done = 0
    # 每图缓存该图的枚举候选（按 core）
    for ii, imname in enumerate(todo_imgs, 1):
        path = os.path.join(DATA, "Images", "visible", imname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as ex:
            print(f"[ERR] open {imname}: {ex!r}", flush=True)
            for k in by_img[imname]:
                direct.setdefault(k, {"bbox": None, "err": "img_open"})
            continue
        # Phase A：直接推理
        for k in by_img[imname]:
            if k in direct:
                continue
            try:
                txt = gen(img, FULL_DIRECT_PROMPT.format(q=q[k]["query"]))
                bs = parse_boxes(txt)
                direct[k] = {"bbox": bs[0] if bs else None, "n": len(bs),
                             "raw": txt[:200]}
            except Exception as ex:
                direct[k] = {"bbox": None, "err": str(ex)[:120]}
            n_done += 1
            if n_done % SAVE_EVERY == 0:
                json.dump(direct, open(OUT_DIRECT, "w", encoding="utf-8"),
                          ensure_ascii=False)
                el = time.time()-t0
                print(f"[A] {ii}/{len(todo_imgs)} 图; 本图批 {n_done} 条 "
                      f"{n_done/el:.2f} q/s", flush=True)

        # Phase B：序数 SoM（按 core 分组枚举一次）
        grp = defaultdict(list)
        for k in by_img[imname]:
            if k in ordinal:
                grp[cores[k].lower()].append(k)
        for core, qids in grp.items():
            cands = None
            try:
                etxt = gen(img, ENUM_PROMPT.format(core=core), max_new=512)
                cands = parse_boxes(etxt)
            except Exception as ex:
                print(f"[ERR] enum {imname}/{core}: {ex!r}", flush=True)
            n_c = len(cands) if cands else 0
            som_img = draw_som(img, cands) if cands else None
            for k in qids:
                if k in som:
                    continue
                rec = {"core": core, "n_cands": n_c}
                if n_c <= 1 or not som_img:
                    rec.update({"num": None, "bbox": None,
                                "reason": "enum_fail"})
                    som[k] = rec
                    continue
                try:
                    stxt = gen(som_img, SEL_PROMPT.format(n=n_c,
                                                          q=q[k]["query"]),
                               max_new=16)
                    m = RE_NUM.search(stxt)
                    num = int(m.group(0)) if m else None
                    if num and 1 <= num <= n_c:
                        rec.update({"num": num, "bbox": cands[num-1],
                                    "bbox_all": cands, "raw": stxt[:80]})
                    else:
                        rec.update({"num": num, "bbox": None,
                                    "bbox_all": cands, "raw": stxt[:80],
                                    "reason": "num_out_of_range"})
                except Exception as ex:
                    rec.update({"num": None, "bbox": None,
                                "reason": "sel_err:" + str(ex)[:80]})
                som[k] = rec
        # 每图落盘
        json.dump(direct, open(OUT_DIRECT, "w", encoding="utf-8"),
                  ensure_ascii=False)
        json.dump(som, open(OUT_SOM, "w", encoding="utf-8"),
                  ensure_ascii=False)
        el = time.time()-t0
        eta = (len(todo_imgs)-ii) * (el/max(ii, 1)) / 3600
        print(f"[{ii}/{len(todo_imgs)}] {imname} direct={len(direct)} "
              f"som={len(som)} 已用{el/3600:.2f}h ETA{eta:.1f}h", flush=True)

    json.dump(direct, open(OUT_DIRECT, "w", encoding="utf-8"),
              ensure_ascii=False)
    json.dump(som, open(OUT_SOM, "w", encoding="utf-8"), ensure_ascii=False)
    nd = sum(1 for v in direct.values() if v.get("bbox"))
    ns = sum(1 for v in som.values() if v.get("bbox"))
    print(f"\n完成 direct {len(direct)}(有框{nd}) som {len(som)}(有框{ns})",
          flush=True)


if __name__ == "__main__":
    main()