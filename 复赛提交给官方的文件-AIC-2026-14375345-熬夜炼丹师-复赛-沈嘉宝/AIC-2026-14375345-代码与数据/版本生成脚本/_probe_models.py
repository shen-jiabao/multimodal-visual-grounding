#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双模型探针：同一批条目、同一套提示词，跑 Qwen3-VL-32B-AWQ 与 8B-Instruct。

目的：为"复赛走 32B-AWQ 还是 8B"提供**实测**依据，而不是凭模型大小的直觉。
对每条序数问句走完整 SoM 三段法（枚举 → 打编号 → 选编号），与线上管线一致。

用法：
    python _probe_models.py --model 32b --n 30
    python _probe_models.py --model 8b  --n 30
输出：/srv/output/probe_<model>.json
"""
import argparse
import base64
import json
import os
import re
import time

from PIL import Image, ImageDraw, ImageFont

DATA = "/srv/dataset"
CORES = "/srv/output/_ordinal_cores_v3.json"
MODEL_8B = "/srv/models/Qwen3-VL-8B-Instruct"
API_BASE = "http://127.0.0.1:8000/v1"
API_MODEL = "qwen3-vl-32b"

# 与 `_qwen_full_pipeline.py` **逐字相同**，保证探针可比。
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
RE_BBOX = re.compile(
    r'"bbox_2d"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]')
RE_NUM = re.compile(r"\d+")


def parse_boxes(text):
    """从模型输出里抽取归一化 bbox。"""
    out = []
    for m in RE_BBOX.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return out


def draw_som(img, boxes):
    """叠加红色编号（矩形宽 3、圆半径 max(12,0.022H)、白字），与线上一致。"""
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
        x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        r = max(12, int(H * 0.022))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0))
        tb = d.textbbox((0, 0), str(i), font=font)
        d.text((cx - (tb[2] - tb[0]) / 2, cy - (tb[3] - tb[1]) / 2 - tb[1]),
               str(i), fill=(255, 255, 255), font=font)
    return im


class Api32B:
    """vLLM OpenAI 兼容接口上的 Qwen3-VL-32B-AWQ。"""
    name = "32b"

    def __init__(self):
        from openai import OpenAI
        self.cli = OpenAI(base_url=API_BASE, api_key="EMPTY")

    def gen(self, img, prompt, max_new=256):
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        b64 = base64.b64encode(buf.getvalue()).decode()
        r = self.cli.chat.completions.create(
            model=API_MODEL, max_tokens=max_new, temperature=0.0,
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt}]}])
        return r.choices[0].message.content


class Hf8B:
    """本地 transformers 上的 Qwen3-VL-8B-Instruct（与初赛同款）。"""
    name = "8b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(MODEL_8B)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_8B, dtype="auto", device_map="auto").eval()

    def gen(self, img, prompt, max_new=256):
        torch = self.torch
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt}]}]
        inp = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=max_new,
                                    do_sample=False)
        return self.proc.batch_decode(
            [o[len(i):] for i, o in zip(inp.input_ids, g)],
            skip_special_tokens=True)[0]


def pick_items(n, qids_file=None, rebuild=False):
    """按固定规则选取 n 条：一半 direct、一半序数。

    两道检查缺一不可：
    ① 图必须在远端存在；
    ② 图必须能被 PIL **完整 load**——上传是逐文件进行的，目标路径上可能
       存在半截文件（`ls` 看得见、`Image.load()` 会抛 "image file is
       truncated"）。本项目已因此崩过一次探针。

    `qids_file`：条目清单。选定后会**写出**该清单，后续模型读同一份。
    这是双模型可比的前提——若各自按"当时可读的图"自行取用，两次条目集
    不同（已被实测证实），比对结论无效。`rebuild=True` 忽略旧清单重新生成，
    但**仍会覆盖写出**（曾漏写导致 32B 无法复用同一清单）。
    """
    q = json.load(open(os.path.join(DATA, "queries", "queries.json"),
                       encoding="utf-8"))
    cores = json.load(open(CORES, encoding="utf-8"))
    if qids_file and os.path.exists(qids_file) and not rebuild:
        sel = json.load(open(qids_file, encoding="utf-8"))
        print(f"沿用既有条目清单 {qids_file}：{len(sel)} 条", flush=True)
        return q, cores, sel

    have = set(os.listdir(os.path.join(DATA, "Images", "visible")))

    def intact(name):
        try:
            Image.open(os.path.join(DATA, "Images", "visible",
                                    name)).load()
            return True
        except Exception:
            return False

    ok = {nm for nm in have if intact(nm)}
    print(f"远端图 {len(have)} 张，可完整读取 {len(ok)} 张", flush=True)
    direct, ordi = [], []
    for k, v in q.items():
        if os.path.basename(v["visible"]) not in ok:
            continue
        (ordi if k in cores else direct).append(k)
    half = n // 2
    sel = direct[:half] + ordi[:half]
    if qids_file:
        json.dump(sel, open(qids_file, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"已写出条目清单 {qids_file}（{len(sel)} 条）", flush=True)
    return q, cores, sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["32b", "8b"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qids", default="/srv/output/probe_qids.json",
                    help="条目清单；首次运行会写出，后续模型沿用同一份")
    ap.add_argument("--rebuild", action="store_true",
                    help="忽略既有清单，按当前可读图重新生成")
    a = ap.parse_args()

    qids_file = a.qids
    q, cores, sel = pick_items(a.n, qids_file, rebuild=a.rebuild)
    print(f"探针条目 {len(sel)} 条（direct+序数），图 {len({q[k]['visible'] for k in sel})} 张",
          flush=True)
    eng = Api32B() if a.model == "32b" else Hf8B()

    res = {}
    imgcache = {}
    out = f"/srv/output/probe_{a.model}.json"

    def save():
        """逐条落盘：探针可能因半截图等意外中断，成果不能只存在内存里。"""
        os.makedirs("/srv/output", exist_ok=True)
        json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1)

    t0 = time.time()
    for i, k in enumerate(sel, 1):
        v = q[k]
        path = os.path.join(DATA, v["visible"])
        try:
            if path not in imgcache:
                imgcache[path] = Image.open(path).convert("RGB")
            img = imgcache[path]
        except Exception as ex:
            print(f"[{i}/{len(sel)}] {k} 跳过（图不可读 {type(ex).__name__}）",
                  flush=True)
            continue
        rec = {"query": v["query"], "img": os.path.basename(path)}
        ts = time.time()
        try:
            txt = eng.gen(img, FULL_DIRECT_PROMPT.format(q=v["query"]))
            bs = parse_boxes(txt)
            rec["direct_bbox"] = bs[0] if bs else None
            rec["direct_nbox"] = len(bs)
            rec["direct_raw"] = txt[:200]
        except Exception as ex:
            rec["direct_err"] = f"{type(ex).__name__}: {ex}"[:200]
        rec["direct_s"] = round(time.time() - ts, 2)

        if k in cores:
            core = cores[k]
            rec["core"] = core
            ts = time.time()
            try:
                etxt = eng.gen(img, ENUM_PROMPT.format(core=core), 512)
                cands = parse_boxes(etxt)
                rec["n_cands"] = len(cands)
                rec["cands"] = cands
                if len(cands) > 1:
                    som = draw_som(img, cands)
                    stxt = eng.gen(som, SEL_PROMPT.format(
                        n=len(cands), q=v["query"]), 16)
                    m = RE_NUM.search(stxt)
                    num = int(m.group(0)) if m else None
                    rec["sel_num"] = num
                    rec["sel_raw"] = stxt[:60]
                    rec["som_bbox"] = (cands[num - 1]
                                       if num and 1 <= num <= len(cands)
                                       else None)
                else:
                    rec["som_bbox"] = None
            except Exception as ex:
                rec["som_err"] = f"{type(ex).__name__}: {ex}"[:200]
            rec["som_s"] = round(time.time() - ts, 2)

        res[k] = rec
        save()
        el = time.time() - t0
        print(f"[{i}/{len(sel)}] {k} {rec['img']} "
              f"direct={rec.get('direct_bbox')} n_cands={rec.get('n_cands')} "
              f"num={rec.get('sel_num')} ({el:.0f}s)", flush=True)

    save()
    print(f"已存 {out}；总耗时 {time.time()-t0:.0f}s；条目 {len(res)}", flush=True)


if __name__ == "__main__":
    main()
