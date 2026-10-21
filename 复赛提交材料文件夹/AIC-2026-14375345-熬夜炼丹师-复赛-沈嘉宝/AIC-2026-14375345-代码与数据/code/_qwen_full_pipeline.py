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
from concurrent.futures import ThreadPoolExecutor, as_completed

# ★ torch / transformers **不在模块级导入**（惰性导入到 hf8b 分支内）。
#   原因：api32b 路线是纯 HTTP 客户端（vLLM 在服务端已加载模型），客户端只需
#   openai + pillow。模块级导入 torch/transformers 会让新服务器被迫装 ~3 GB 的
#   CUDA 版 torch——而这条路线根本用不到。
from PIL import Image, ImageDraw, ImageFont
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

MODEL_PATH = os.environ.get("QWEN_MODEL", "/srv/models/Qwen3-VL-8B-Instruct")
DATA = os.environ.get("QWEN_DATA", "/srv/dataset")
OUT_DIRECT = os.environ.get("QWEN_OUT_DIRECT", "/srv/output/qwen3vl_all.json")
OUT_SOM = os.environ.get("QWEN_OUT_SOM", "/srv/output/qwen3vl_som_all.json")
CORES = os.environ.get("QWEN_CORES", "/srv/output/_ordinal_cores_v3.json")
SAVE_EVERY = 50

# --------------------- 引擎选择（2026-09-25 追加，用于复赛"换更强模型"实验）---------------------
# hf8b  （默认）= 原线上行为：本地 transformers 上的 Qwen3-VL-8B-Instruct。默认值保证
#        初赛 0.7827 与复赛 0.7459 的产物仍可逐字节复现。
# api32b        = 走 vLLM 的 OpenAI 兼容接口上的 Qwen3-VL-32B-AWQ（探针已证其在"第 k 个
#        实例"上强于 8B：8B 给重复框、32B 给等间距不重复框）。
# ★换模型时**提示词必须一字不改**——否则模型差异与提示词差异混在一起，无法归因。故
#   FULL_DIRECT_PROMPT / ENUM_PROMPT / SEL_PROMPT 三个常量对本开关完全无感。
ENGINE = os.environ.get("QWEN_ENGINE", "hf8b")
API_BASE = os.environ.get("QWEN_API_BASE", "http://127.0.0.1:8000/v1")
API_MODEL = os.environ.get("QWEN_API_MODEL", "qwen3-vl-32b")
CONC = int(os.environ.get("QWEN_CONC", "1"))        # >1 时 Phase A 并发（仅 api 引擎生效）
ONLY_NONORD = os.environ.get("QWEN_ONLY_NONORD", "0") == "1"   # 只跑非序数（省算力）
PHASE = os.environ.get("QWEN_PHASE", "AB")          # A / B / AB
# --------------------- 三模态开关（2026-09-29 追加，用于预登记 D）---------------------
# "1"（默认）= 原线上行为：只读可见光图。默认值保证既有产物仍可逐字节复现。
# "3"        = 同一模型的**同一次 direct 定位**，但输入增加红外与深度两张图
#              （预登记 D §3 的固定图序与预处理）。**输出约定一字不改**——见下方断言。
MODALITIES = os.environ.get("QWEN_MODALITIES", "1")
# 可选：只跑这些 qid（Phase A）。用于小样与"按作用范围"的全量（预登记 D §1）。
QIDS_FILE = os.environ.get("QWEN_QIDS", "")

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

# --------------------- 三模态输入（预登记 D §3）---------------------
# 与 FULL_DIRECT_PROMPT 的唯一差别是**输入描述**（三图顺序与各图含义）；
# 描述之后的部分必须逐字相同。下面这条断言就是"输出约定未变"的机器证明——
# 一旦有人改动其中一边，脚本会立刻失败而不是悄悄改变机制。
THREE_IMG_PROMPT = (
    "The three images below show the same scene in this fixed order: "
    "(1) visible-light colour photo, (2) thermal infrared (brighter = hotter), "
    "(3) depth map (brighter = farther, black = invalid depth).\n"
    "Locate the object described below and output its bounding box in the "
    "visible-light image (image 1).\nDescription: {q}\n"
    "Output ONLY a JSON array like "
    '[{{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}] '
    "where coordinates are integers normalized to 0-1000 "
    "(x1,y1 = top-left, x2,y2 = bottom-right)."
)
assert (THREE_IMG_PROMPT.split("{q}")[1]
        == FULL_DIRECT_PROMPT.split("{q}")[1]), \
    "三图提示词的输出约定与单图提示词不一致（必须逐字相同）"

DEPTH_MAX = 20000.0        # 深度有效上界（mm）；实测数据 max=19999


def load_ir(path):
    """红外：三通道**高度一致**（实测通道间 |Δ| ≤ 1，为 1/255 量化差）
    ⇒ 取单通道转灰度即可，不丢信息。"""
    return Image.open(path).convert("L").convert("RGB")


def load_depth(path, dmax=DEPTH_MAX):
    """深度预处理（预登记 D §3 + 数据格式补遗，见跑后实录 §十一）。

    实测数据集有**两个子群**，深度的存储格式不同（赛题文档只描述了前者）：

    · **.png 组**（被引用的 1295 张中占 1198）：`I;16` 单通道 uint16、单位毫米、
      实测 min=0 max=19999 —— 按 D §3 原规则处理：
      裁 [0,20000] → 线性拉伸为 8 位灰度 → 无效区（原值 0）置黑。

    · **.jpg 组**（占 97 张，visible/infrared/depth 均为 JPEG 640×360）：
      深度是 `RGB` 三通道，且**三通道逐像素相同**（实测 |Δ| = 0）⇒ 实为**8 位灰度**；
      JPEG 有损，原始 16 位量程已被破坏、无法还原。
      此时**取单通道原值直通**，不按任何比例缩放——因为赛题语义只需"越亮越远"的
      **单调性**，8 位原值已满足；按未知比例换算成毫米等于凭空造信息。

    两个子群**内部**三模态同尺寸（已实测：.png 组均 1920×1080、.jpg 组均 640×360），
    故跨模态对齐性不受影响。
    """
    import numpy as np
    a = np.asarray(Image.open(path))
    if a.ndim == 3:                       # .jpg 组：8 位灰度被复制成三通道
        return Image.fromarray(a[..., 0], mode="L").convert("RGB")
    arr = a.astype("float32")             # .png 组：16 位毫米
    g = np.clip(arr, 0.0, dmax) / dmax * 255.0
    out = g.round().astype("uint8")
    out[arr <= 0] = 0                     # 无效深度区 → 黑
    return Image.fromarray(out, mode="L").convert("RGB")


# ===== 输出适配：bbox 的坐标空间 =====
# 本项目提示词要求"整数、归一化到 0-1000"，但**不是所有模型都遵守**：
#   · Qwen3-VL 系（8B / 32B / 3.8-27B）：遵守 ⇒ 除以 1000（缺省，与改造前逐位相同）
#   · Qwen2.5-VL 系（72B-Instruct-AWQ）：输出**它自己 smart-resize 之后的像素坐标**
#     （边长取整到 28 的倍数）。若仍按 ÷1000 解读，后果分两种、都很坏：
#       ① 大图坐标越界（>1.0）⇒ 装配器判"框非法"⇒ 该条保留基线（浪费但无害）
#       ② 中小坐标**不越界但整体错位** ⇒ 被当作合法改动接受 ⇒ **整批错误框**
#     ②是最坏形态（"整批完全失效"那一类），而 G2 的"可解析"判据抓不到它。
# 判据来源：20 条冒烟 + **交付件框作独立参照**（`_k_scale_diag.py`），
# 只比几何一致性、**不看分数** ⇒ 不是事后调参，是读出模型的真实输出约定。
BBOX_SPACE = os.environ.get("QWEN_BBOX_SPACE", "norm1000")


def smart_resize_wh(w, h, factor=28, min_px=4 * 28 * 28,
                    max_px=16384 * 28 * 28):
    """复刻 Qwen2.5-VL 的 smart_resize 目标边长（不真的缩放图像）。

    与 transformers 的 Qwen2VLImageProcessor 同规则：先把边长取整到 factor 的
    倍数；总像素越界时按面积比例整体缩放后再取整。本项目图像（1920x1080 与
    640x360 两个子群）都落在区间内 ⇒ 走第一条分支。
    """
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_px:
        beta = (h * w / max_px) ** 0.5
        hb = max(factor, int(h / beta / factor) * factor)
        wb = max(factor, int(w / beta / factor) * factor)
    elif hb * wb < min_px:
        beta = (min_px / (h * w)) ** 0.5
        hb = max(factor, int(h * beta / factor) * factor)
        wb = max(factor, int(w * beta / factor) * factor)
    return wb, hb


def parse_boxes(text, img_wh=None):
    """解析模型输出的 bbox 整数并按 BBOX_SPACE 归一化。

    img_wh：原图 (W, H)，仅在 BBOX_SPACE == "smart_resize_px" 时使用。
    缺省路径（norm1000）与改造前**逐位相同** ⇒ 既有各件行为不受影响。
    """
    out = []
    for m in RE_BBOX.finditer(text):
        x1, y1, x2, y2 = map(int, m.groups())
        if BBOX_SPACE == "smart_resize_px" and img_wh:
            sw, sh = smart_resize_wh(img_wh[0], img_wh[1])
            v = [min(max(x1 / sw, 0.0), 1.0), min(max(y1 / sh, 0.0), 1.0),
                 min(max(x2 / sw, 0.0), 1.0), min(max(y2 / sh, 0.0), 1.0)]
            # 模型会有小幅越界（实测 x2 到过 2000 vs 边界 1932）⇒ 裁剪 + 角点排序
            out.append([min(v[0], v[2]), min(v[1], v[3]),
                        max(v[0], v[2]), max(v[1], v[3])])
        else:
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
        # 新版 PIL 不再容忍逆序矩形（x1>x2 直接抛 ValueError；实测已崩过一次
        # Phase B，停在 119/545）。这里**只对绘制用的矩形**做角点排序——候选表
        # `boxes` 与 `num → cands[num-1]` 的映射都不经过此处，故编号与候选的
        # 对应关系逐位不变，属于"渲染层兼容性修复"，不改机制。
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        cx, cy = (x1+x2)/2, (y1+y2)/2
        r = max(12, int(H*0.022))
        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(255, 0, 0))
        tb = d.textbbox((0, 0), str(i), font=font)
        d.text((cx-(tb[2]-tb[0])/2, cy-(tb[3]-tb[1])/2 - tb[1]),
               str(i), fill=(255, 255, 255), font=font)
    return im


def make_gen_api():
    """构造 vLLM(OpenAI 兼容) 引擎的 gen。**模块级**——烟测脚本 `_smoke_qwen32b.py`
    直接复用它，避免"两份 gen 实现"漂移（本项目纪律：判据只实现一处）。

    编码口径与 `_probe_models.py` 的 Api32B 完全相同（JPEG quality=95 + base64 data URI），
    这是烟测能与 `probe_32b.json` 逐条比对的前提。
    """
    import base64
    import io

    from openai import OpenAI
    cli = OpenAI(base_url=API_BASE, api_key="EMPTY")

    def gen(img, prompt, max_new=256):
        # img 可为单张图（单模态）或图列表（三模态，预登记 D）。单张时请求体与
        # 改造前**逐字节相同**——故既有 32B 基线的行为不受本改动影响。
        imgs = img if isinstance(img, (list, tuple)) else [img]
        content = []
        for im in imgs:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=95)
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + b64}})
        content.append({"type": "text", "text": prompt})
        r = cli.chat.completions.create(
            model=API_MODEL, max_tokens=max_new, temperature=0.0,
            messages=[{"role": "user", "content": content}])
        return r.choices[0].message.content

    return gen


def _direct_one(gen, img, query):
    """单条直接定位。抽成独立函数是为了让 Phase A 能并发（api 引擎下 vLLM 侧
    max_num_seqs=8，串行会浪费整条流水线）；异常就地兜住，返回 error 记录。

    `img` 可为单张图或图列表（三模态）；提示词随 MODALITIES 切换，但两者的
    **输出约定逐字相同**（模块级断言已保证）。"""
    prompt = (THREE_IMG_PROMPT if MODALITIES == "3"
              else FULL_DIRECT_PROMPT).format(q=query)
    try:
        txt = gen(img, prompt)
        # 坐标空间适配需要原图尺寸（见 BBOX_SPACE 注释）；多图取可见光那张
        if isinstance(img, (list, tuple)):
            wh = img[0].size if img else None
        else:
            wh = img.size
        bs = parse_boxes(txt, wh)
        return {"bbox": bs[0] if bs else None, "n": len(bs), "raw": txt[:200]}
    except Exception as ex:
        return {"bbox": None, "err": str(ex)[:120]}


def main():
    os.makedirs(os.path.dirname(OUT_DIRECT), exist_ok=True)
    q = json.load(open(os.path.join(DATA, "queries", "queries.json"),
                       encoding="utf-8"))
    cores = json.load(open(CORES, encoding="utf-8")) if os.path.exists(CORES) else {}
    ordinal = set(cores)
    # 可选作用范围（预登记 D §1）：只跑这些 qid 的 Phase A
    only_q = None
    if QIDS_FILE and os.path.exists(QIDS_FILE):
        only_q = set(json.load(open(QIDS_FILE, encoding="utf-8")))
        miss = [k for k in only_q if k not in q]
        assert not miss, f"QWEN_QIDS 里有 {len(miss)} 个 qid 不在 queries.json"

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
    print(f"[pipe] 模态={MODALITIES} 作用范围={'全部' if only_q is None else str(len(only_q)) + ' 条'}"
          f" 提示词={'三图' if MODALITIES == '3' else '单图'}", flush=True)
    print(f"已完成 direct={len(direct)} som={len(som)}", flush=True)

    # 引擎装配。★api32b 分支**绝不加载 8B 权重**——否则会与 vLLM 争显存。
    if ENGINE == "api32b":
        gen = make_gen_api()
        print(f"[pipe] 引擎 api32b → {API_BASE} model={API_MODEL} "
              f"并发={CONC} 仅非序数={ONLY_NONORD} 阶段={PHASE}", flush=True)
    else:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype="auto", device_map="auto").eval()
        print(f"[pipe] 引擎 hf8b 模型就绪 显存 "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        def gen(img, prompt, max_new=256):
            imgs = img if isinstance(img, (list, tuple)) else [img]
            msgs = [{"role": "user", "content": (
                [{"type": "image", "image": im} for im in imgs]
                + [{"type": "text", "text": prompt}])}]
            inp = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=max_new,
                                   do_sample=False)
            return processor.batch_decode(
                [o[len(i):] for i, o in zip(inp.input_ids, g)],
                skip_special_tokens=True)[0]

    # 断点：direct 未齐 或 序数 SoM 未齐 的图都要处理。
    # 注意：两个阶段内部各自幂等（direct 命中即跳过、som 命中即跳过），
    # 因此二次进入同一张图只会补齐缺失的那部分，不会重算。
    # ★必须与 PHASE / ONLY_NONORD 口径一致，否则被跳过的部分会让"待处理图"永远非空
    #   （跑完也退不出循环）。
    def _pending(im):
        if PHASE in ("A", "AB"):
            for k in by_img[im]:
                if only_q is not None and k not in only_q:
                    continue
                if k in direct:
                    continue
                if ONLY_NONORD and k in ordinal:
                    continue
                return True
        if PHASE in ("B", "AB"):
            return any(k in ordinal and k not in som for k in by_img[im])
        return False

    todo_imgs = [im for im in imgs if _pending(im)]
    print(f"待处理图 {len(todo_imgs)}", flush=True)

    t0 = time.time()
    n_done = 0
    n_saved = 0
    # 每图缓存该图的枚举候选（按 core）
    for ii, imname in enumerate(todo_imgs, 1):
        path = os.path.join(DATA, "Images", "visible", imname)
        try:
            img = Image.open(path).convert("RGB")     # 可见光（Phase B 的 SoM 底图）
            if MODALITIES == "3":
                # 预登记 D §3 的固定图序：1=可见光  2=红外  3=深度
                img_in = [img,
                          load_ir(os.path.join(DATA, "Images", "infrared", imname)),
                          load_depth(os.path.join(DATA, "Images", "depth", imname))]
            else:
                img_in = img
        except Exception as ex:
            print(f"[ERR] open {imname}: {ex!r}", flush=True)
            for k in by_img[imname]:
                # ★必须与 Phase A 同口径过滤，否则会把作用范围外的条目也写进产物
                #   （实测：9 张图报 img_open 时带进了 16 条清单外条目）
                if only_q is not None and k not in only_q:
                    continue
                direct.setdefault(k, {"bbox": None, "err": "img_open"})
            continue
        # Phase A：直接推理
        pend = []
        if PHASE in ("A", "AB"):
            for k in by_img[imname]:
                if only_q is not None and k not in only_q:
                    continue
                if k in direct:
                    continue
                if ONLY_NONORD and k in ordinal:
                    continue
                pend.append(k)
        if CONC > 1 and ENGINE == "api32b" and len(pend) > 1:
            # 并发路径：仅 api 引擎（HTTP 无状态）。结果按 qid 写回，顺序无关。
            with ThreadPoolExecutor(max_workers=CONC) as ex:
                futs = {ex.submit(_direct_one, gen, img_in, q[k]["query"]): k
                        for k in pend}
                for fu in as_completed(futs):
                    direct[futs[fu]] = fu.result()
                    n_done += 1
        else:
            for k in pend:
                direct[k] = _direct_one(gen, img_in, q[k]["query"])
                n_done += 1
        if n_done - n_saved >= SAVE_EVERY:
            json.dump(direct, open(OUT_DIRECT, "w", encoding="utf-8"),
                      ensure_ascii=False)
            n_saved = n_done
            el = time.time()-t0
            print(f"[A] {ii}/{len(todo_imgs)} 图; 已出 {n_done} 条 "
                  f"{n_done/el:.2f} q/s", flush=True)

        # Phase B：序数 SoM（按 core 分组枚举一次）
        grp = defaultdict(list)
        if PHASE in ("B", "AB"):
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