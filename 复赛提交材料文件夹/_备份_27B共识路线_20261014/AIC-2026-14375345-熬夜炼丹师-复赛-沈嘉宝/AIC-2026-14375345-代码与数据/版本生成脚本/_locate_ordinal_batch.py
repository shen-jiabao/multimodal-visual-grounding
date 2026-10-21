#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""序数层推广：对 7 张新图各做一次 P3 全候选枚举（LocateAnything-3B slow）。

- 同一图片的多个序数问题共享同一次枚举候选池。
- 每张图推理前检查空闲显存（阈值 12GiB）；被其他会话抢占时轮询等待，
  超过 20 分钟仍不可用则落 checkpoint 并退出（可续跑）。
- 每张图独立 try/except + 即时写 checkpoint，单图失败不拖垮全批。
- 仅产出模型原始输出与全部框；去重/排序/取第 N 在本地用
  locate_ordinal_select.py 完成，保证规则与探针完全一致。
"""
import json
import os
import re
import subprocess
import time

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoConfig

MODEL_PATH = "/models/LocateAnything-3B/LocateAnything-3B"
DATA_ROOT = "/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理"
MANIFEST = os.environ.get("BATCH_MANIFEST", "/root/ordinal_batch_manifest.json")
OUT_JSON = os.environ.get("BATCH_OUT", "/root/output/locate_ordinal_batch.json")
MAX_IMAGE_SIZE = 2500
MAX_NEW_TOKENS = 1024  # 密集人群可能需要更多 token
FREE_VRAM_MIB = 12000
WAIT_ROUNDS = 60       # 60 * 20s = 20 分钟

BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

# 不规则复数
IRREGULAR_PLURAL = {"person": "people", "child": "children", "man": "men",
                    "woman": "women"}


def pluralize(phrase: str) -> str:
    """名词短语的简单复数（仅复数化最后一个实词）。"""
    words = phrase.strip().split()
    last = words[-1]
    low = last.lower()
    if low in IRREGULAR_PLURAL:
        words[-1] = IRREGULAR_PLURAL[low]
    elif low.endswith(("s", "x", "z", "ch", "sh")):
        words[-1] = last + "es"
    elif low.endswith("y") and low[-2] not in "aeiou":
        words[-1] = last[:-1] + "ies"
    else:
        words[-1] = last + "s"
    return " ".join(words)


def build_prompt(core: str) -> str:
    """P3 枚举 prompt：先报总数，再从左到右逐框，含边缘/遮挡实例。"""
    plural = pluralize(core)
    return (
        f"How many {plural} are there in this image? First state the count as a "
        f"single number, then output one <box><x1><y1><x2><y2></box> for EACH "
        f"{core} from left to right. Include every instance, including ones "
        f"partially cut off by the image edge and ones occluded by other objects."
    )


def parse_boxes(answer: str):
    """解析全部归一化框（0-1000 → 0-1），保持发射顺序。"""
    out = []
    for m in BOX_RE.finditer(answer):
        x1, y1, x2, y2 = map(int, m.groups())
        out.append([x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0])
    return out


def gpu_free_mib() -> int:
    """查询当前空闲显存（MiB）；查询失败返回 0。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return 0


def wait_for_gpu() -> bool:
    """【仅模型加载前】确认总空闲显存；被其他会话占满则轮询，超时返回 False。"""
    for i in range(WAIT_ROUNDS):
        free = gpu_free_mib()
        print(f"    [GPU] free={free}MiB (round {i+1}/{WAIT_ROUNDS})", flush=True)
        if free >= FREE_VRAM_MIB:
            return True
        time.sleep(20)
    return False


def gpu_compute_pids():
    """返回 nvidia-smi 报告的 GPU 计算进程宿主 PID 集合。

    容器内 PID 与宿主 PID 不同，无法与本地 pgrep 比对；但"新出现的 PID"
    即代表其他会话抢卡，用集合差即可可靠识别。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader"], text=True)
        return {int(x.strip()) for x in out.splitlines() if x.strip().isdigit()}
    except Exception:
        return set()


def wait_no_foreign(baseline_pids, rounds=WAIT_ROUNDS):
    """【模型加载后】仅当出现基线之外的新 GPU 进程时等待。

    slow 模式自身显存会随图像在 8~32GB 间大幅波动，绝不能用显存余量判断；
    只认"新的计算进程 PID"。
    """
    for i in range(rounds):
        cur = gpu_compute_pids()
        foreign = cur - baseline_pids
        print(f"    [GPU] pids={sorted(cur)} foreign={sorted(foreign)} "
              f"(round {i+1}/{rounds})", flush=True)
        if not foreign:
            return True
        time.sleep(20)
    return False


def main():
    """逐图枚举并写 checkpoint。"""
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    results = {}
    if os.path.exists(OUT_JSON):
        results = json.load(open(OUT_JSON, encoding="utf-8"))
        print(f"续跑：已有 {len(results)} 张图结果", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 加载前：要求总空闲显存足够（此时显存里还没有我们自己的模型）
    if not wait_for_gpu():
        print("启动等待超时：GPU 被占用，未启动（无副作用）。", flush=True)
        return
    print("加载 LocateAnything（overlay 4.52.4 / sdpa）...", flush=True)
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
        trust_remote_code=True).to(device).eval()
    print("模型就绪", flush=True)
    # 记录我方 GPU 计算进程宿主 PID 集合；之后只对"新 PID"判抢占
    # （slow 模式自身显存 8~32GB 波动，不能用显存余量判断）
    torch.cuda.synchronize()
    baseline_pids = gpu_compute_pids()
    print(f"自身 GPU PID 集合 baseline_pids={sorted(baseline_pids)}", flush=True)

    items = [(img, info) for img, info in manifest.items()
             if info.get("tag", img) not in results]
    print(f"待处理图片 {len(items)} 张", flush=True)

    for idx, (img_id, info) in enumerate(items, start=1):
        tag = info.get("tag", img_id)
        prompt = info.get("prompt") or build_prompt(info["core"])
        print(f"\n=== [{idx}/{len(items)}] {tag} core={info['core']!r} ===",
              flush=True)
        if not wait_no_foreign(baseline_pids):
            print("检测到外来进程持续占卡，保存 checkpoint 后退出（可重跑续跑）。",
                  flush=True)
            break
        try:
            image = Image.open(os.path.join(DATA_ROOT, info["visible"])).convert("RGB")
            w0, h0 = image.size
            if max(w0, h0) > MAX_IMAGE_SIZE:
                s = MAX_IMAGE_SIZE / max(w0, h0)
                image = image.resize((int(w0 * s), int(h0 * s)), Image.LANCZOS)

            print(f"    prompt: {prompt[:120]}...", flush=True)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}]}]
            text = processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, _ = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images,
                               return_tensors="pt").to(device)
            with torch.no_grad():
                resp = model.generate(
                    pixel_values=inputs["pixel_values"].to(model.dtype),
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws"),
                    tokenizer=tokenizer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    generation_mode="slow",
                    do_sample=False,
                    verbose=False)
            if isinstance(resp, tuple):
                answer = resp[0][0] if isinstance(resp[0], list) else resp[0]
            else:
                answer = resp[0] if isinstance(resp, list) else resp
            answer = str(answer)
            boxes = parse_boxes(answer)
            results[tag] = {
                "img": img_id,
                "visible": info["visible"],
                "core": info["core"],
                "prompt": prompt,
                "queries": info["queries"],
                "raw": answer,
                "n_raw_boxes": len(boxes),
                "boxes": boxes,
            }
            print(f"    OK raw_boxes={len(boxes)}", flush=True)
        except Exception as exc:  # 单图失败隔离
            results[tag] = {"error": repr(exc), "img": img_id,
                            "visible": info["visible"],
                            "core": info["core"], "queries": info["queries"]}
            print(f"    FAIL {exc!r}", flush=True)

        # 即时 checkpoint
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    print(f"\n完成，结果写入 {OUT_JSON}（{len(results)} 张图）", flush=True)


if __name__ == "__main__":
    main()