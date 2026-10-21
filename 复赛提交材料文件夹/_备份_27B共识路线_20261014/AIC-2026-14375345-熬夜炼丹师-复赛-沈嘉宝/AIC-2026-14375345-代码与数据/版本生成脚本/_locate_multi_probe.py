#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探针：验证 LocateAnything-3B 能否用一个 core 短语一次输出全部同类实例框。

只跑图 000023（stone pier 序数三连所在图），尝试多种"列出全部"prompt，
打印模型原始输出 + 解析出的全部归一化框及其面积/中心 x。
不写任何提交文件，纯探测。
"""
import os
import re
import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig

MODEL_PATH = "/models/LocateAnything-3B/LocateAnything-3B"
DATA_ROOT = "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理"
IMG_REL = "Images/visible/000023.png"
MAX_IMAGE_SIZE = 2500
MAX_NEW_TOKENS = 512  # 多框需要更多 token

# 多种"列出全部实例"的 prompt 措辞，探测哪种能让模型吐出多个 <box>
PROMPTS = [
    ("P1_all_l2r",
     "Locate ALL stone piers in this image. Output EVERY stone pier as a separate "
     "box in the format <box><x1><y1><x2><y2></box>, ordered from left to right. "
     "Include stone piers that are partially cut off by the image edge."),
    ("P2_every_enum",
     "List every spherical stone bollard (stone pier) visible in the image, one box "
     "each. Enumerate them from the far left to the far right and output a "
     "<box><x1><y1><x2><y2></box> for each, including partial ones at the border."),
    ("P3_count_then_box",
     "How many stone piers are there in this image? First state the count, then output "
     "one <box><x1><y1><x2><y2></box> for each stone pier from left to right, "
     "including any partially visible at the edge."),
]

BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer):
    """解析模型输出中的全部归一化框（0-1000 → 0-1），保持出现顺序。"""
    out = []
    for m in BOX_RE.finditer(answer):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return out


def box_stat(b):
    """返回面积与中心坐标，便于核对排序。"""
    area = (b[2] - b[0]) * (b[3] - b[1])
    cx = (b[0] + b[2]) / 2.0
    cy = (b[1] + b[3]) / 2.0
    return area, cx, cy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("加载 tokenizer/processor/model ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, fix_mistral_regex=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True)
    # 厂商 __init__ 优先读 text_config._attn_implementation（否则落到自定义 magi/eager），
    # 顶层 attn_implementation kwarg 传不进内层 Qwen2，故构造前显式写子配置。
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    if getattr(config, "vision_config", None) is not None:
        config.vision_config._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(
        MODEL_PATH, config=config, torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device).eval()
    print("文本 attn =",
          model.language_model.config._attn_implementation, flush=True)
    print("模型就绪", flush=True)

    img = Image.open(os.path.join(DATA_ROOT, IMG_REL)).convert("RGB")
    w0, h0 = img.size
    if max(w0, h0) > MAX_IMAGE_SIZE:
        s = MAX_IMAGE_SIZE / max(w0, h0)
        img = img.resize((int(w0 * s), int(h0 * s)), Image.LANCZOS)
    print(f"图像 {IMG_REL} 原始 {w0}x{h0}", flush=True)

    results = {}
    for name, prompt in PROMPTS:
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
        boxes = parse_boxes(answer)
        results[name] = {"prompt": prompt, "raw": answer, "boxes": boxes}

        print("\n" + "=" * 90)
        print(f"[{name}] 解析到 {len(boxes)} 个框")
        print(f"RAW: {answer.strip()[:500]}")
        # 按模型输出顺序打印
        for i, b in enumerate(boxes):
            area, cx, cy = box_stat(b)
            print(f"  out#{i:02d} box={[round(v,3) for v in b]} "
                  f"area={area:.4f} cx={cx:.3f} cy={cy:.3f}")
        # 再按中心 x 稳定排序打印（方案 2 的取法）
        order = sorted(range(len(boxes)),
                       key=lambda i: (box_stat(boxes[i])[1],
                                      box_stat(boxes[i])[2], i))
        print("  -- 按 cx 升序 --")
        for rank, i in enumerate(order, start=1):
            area, cx, cy = box_stat(boxes[i])
            print(f"  rank{rank} <- out#{i:02d} cx={cx:.3f} area={area:.4f} "
                  f"box={[round(v,3) for v in boxes[i]]}")

    with open("/root/output/locate_multi_probe_000023.json", "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n已保存 /root/output/locate_multi_probe_000023.json", flush=True)


if __name__ == "__main__":
    main()