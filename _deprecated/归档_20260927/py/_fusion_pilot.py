# -*- coding: utf-8 -*-
"""分歧子集(235条) CLIP 试融合：
规则：跨家族 IoU>=0.5 -> 0.6*Locate + 0.4*GD 加权平均；否则 CLIP 图文相似度选优；
      先剔除面积>0.5 的候选（若全是大框则保留）。
输出：/root/output/fusion_pilot_235.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(4)

DATA = Path("/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理")
OUT = Path("/root/output")
DETAIL = OUT / "fusion_detail.json"

NAMES = {"L2": "L2_locate_v2_hires", "M": "M_gd3modal", "G": "G_groundingdino"}


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def crop(img, box, pad=0.05):
    W, H = img.size
    x1, y1, x2, y2 = box
    bw, bh = (x2-x1)*W, (y2-y1)*H
    cx1 = max(0, int(x1*W - bw*pad)); cy1 = max(0, int(y1*H - bh*pad))
    cx2 = min(W, int(x2*W + bw*pad)); cy2 = min(H, int(y2*H + bh*pad))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img.crop((cx1, cy1, cx2, cy2))


def main():
    detail = json.loads(DETAIL.read_text(encoding="utf-8"))
    queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
    todos = [k for k, v in detail.items() if v["max_iou"] < 0.5]
    print(f"分歧样本: {len(todos)} 条")

    import open_clip
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    print(f"CLIP 就绪 {time.time()-t0:.0f}s (CPU)")

    results = {}
    stats = {"weighted_pair": 0, "clip": 0, "area_filtered": 0,
             "clip_pick": {"L2": 0, "M": 0, "G": 0}}
    for n, qid in enumerate(todos, 1):
        rec = queries.get(qid) or {}
        img_path = DATA / rec.get("visible", "")
        text = rec.get("query", "")
        boxes = {k: detail[qid]["boxes"][v] for k, v in NAMES.items()}
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[skip] {qid} 图像失败 {e}")
            continue

        # 1) 面积过滤
        elig = {k: b for k, b in boxes.items() if area(b) <= 0.5}
        if not elig:
            elig = dict(boxes)
        else:
            stats["area_filtered"] += sum(1 for k in boxes if k not in elig)

        # 2) 跨家族 IoU>=0.5 -> 加权平均
        keys = list(elig)
        pair, best = None, 0.0
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                v = iou(elig[keys[i]], elig[keys[j]])
                if v > best:
                    best, pair = v, (keys[i], keys[j])
        if pair and best >= 0.5:
            a_k, b_k = pair
            wa = 0.6 if a_k == "L2" else (0.4 if b_k == "L2" else 0.5)
            wb = 1.0 - wa
            ba, bb = elig[a_k], elig[b_k]
            fused = [wa*ba[i] + wb*bb[i] for i in range(4)]
            results[qid] = {"bbox": [round(v, 6) for v in fused],
                            "source": f"weighted:{a_k}+{b_k}", "iou": round(best, 4)}
            stats["weighted_pair"] += 1
        else:
            # 3) CLIP 选优
            crops = [(k, crop(img, b)) for k, b in elig.items()]
            crops = [(k, c) for k, c in crops if c is not None]
            if not crops:
                continue
            batch = torch.stack([preprocess(c) for _, c in crops])
            txt = tokenizer([text])
            with torch.no_grad():
                fi = model.encode_image(batch)
                ft = model.encode_text(txt)
                fi = fi / fi.norm(dim=-1, keepdim=True)
                ft = ft / ft.norm(dim=-1, keepdim=True)
                sims = (fi @ ft.T).squeeze(-1).tolist()
            bi = int(np.argmax(sims))
            pick = crops[bi][0]
            results[qid] = {"bbox": [round(float(v), 6) for v in elig[pick]],
                            "source": f"clip:{pick}",
                            "sim": {k: round(float(s), 4) for (k, _), s in zip(crops, sims)}}
            stats["clip"] += 1
            stats["clip_pick"][pick] += 1
        if n % 50 == 0:
            print(f"  [{n}/{len(todos)}]")

    out_p = OUT / "fusion_pilot_235.json"
    out_p.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n统计:", stats)
    print("输出:", out_p, f"{out_p.stat().st_size/1024:.0f} KB")

    # 与三个单模型的一致率
    print("\n融合结果 vs 单模型 (IoU>=0.5 占比):")
    for k, nk in NAMES.items():
        c = sum(1 for q, r in results.items()
                if iou(r["bbox"], detail[q]["boxes"][nk]) >= 0.5)
        print(f"  {k}: {c}/{len(results)} ({100*c/len(results):.1f}%)")


if __name__ == "__main__":
    main()
