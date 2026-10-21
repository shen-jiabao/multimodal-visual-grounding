# -*- coding: utf-8 -*-
"""30 条分层 sanity check：对 L2/M/G 三个候选框做 CLIP 打分并输出判读指标。
分层：简单(颜色/单目标) / 空间(方位词) / 困难(序数·计数)
输出：/root/output/sanity30.json
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

torch.set_num_threads(4)

DATA = Path("/root/dataset/初赛数据集-基于大模型的多模态视觉理解与推理")
OUT = Path("/root/output")

HARD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b"
                  r"|\b(two|three|four|five|six|seven|eight|nine|ten|pair|both)\b"
                  r"|from (the )?(left|right) to (the )?(left|right)", re.I)
MED = re.compile(r"leftmost|rightmost|topmost|bottommost|\bleft\b|\bright\b|above|below|behind"
                 r"|in front|top of|bottom of|beside|next to|\bnear\b|closest|farthest"
                 r"|foreground|background|between|upper|lower", re.I)
COLOR = re.compile(r"red|blue|green|yellow|black|white|gray|grey|brown|orange|pink|purple"
                   r"|silver|gold|teal|beige", re.I)
CJK = re.compile(r"[\u4e00-\u9fff]")


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2-x1); ih = max(0.0, y2-y1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def area(b):
    return (b[2]-b[0])*(b[3]-b[1])


def crop(img, box, pad=0.06):
    W, H = img.size
    x1, y1, x2, y2 = box
    bw, bh = (x2-x1)*W, (y2-y1)*H
    cx1 = max(0, int(x1*W - bw*pad)); cy1 = max(0, int(y1*H - bh*pad))
    cx2 = min(W, int(x2*W + bw*pad)); cy2 = min(H, int(y2*H + bh*pad))
    return None if (cx2 <= cx1 or cy2 <= cy1) else img.crop((cx1, cy1, cx2, cy2))


def main():
    queries = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
    det = json.loads((OUT / "fusion_detail.json").read_text(encoding="utf-8"))

    strata = {"easy": [], "medium": [], "hard": []}
    seen_scene = {k: set() for k in strata}
    for k, rec in queries.items():
        q = rec["query"]
        if CJK.search(q) or k not in det:
            continue
        scene = k.split("_")[0]
        if HARD.search(q):
            s = "hard"
        elif MED.search(q):
            s = "medium"
        elif COLOR.search(q):
            s = "easy"
        else:
            continue
        if scene in seen_scene[s] or len(strata[s]) >= 10:
            continue
        seen_scene[s].add(scene)
        strata[s].append(k)

    print({k: len(v) for k, v in strata.items()})

    import open_clip
    model, _, pre = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True)
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    KEYS = {"L2": "L2_locate_v2_hires", "M": "M_gd3modal", "G": "G_groundingdino"}
    out = {}
    for s in ("easy", "medium", "hard"):
        print("\n" + "=" * 118)
        print(f"### {s.upper()} ({len(strata[s])} 条)")
        print(f"{'qid':14s} {'margin':>7s} {'simL2':>7s} {'simM':>7s} {'simG':>7s} "
              f"{'IoU_LM':>7s} {'IoU_LG':>7s} {'IoU_MG':>7s} {'top1':>5s}  query")
        for k in strata[s]:
            q = queries[k]["query"]
            boxes = {t: det[k]["boxes"][kk] for t, kk in KEYS.items()}
            try:
                img = Image.open(DATA / queries[k]["visible"]).convert("RGB")
            except Exception as e:
                print(f"  {k} 图像失败 {e}")
                continue
            crops, tags = [], []
            for t, b in boxes.items():
                c = crop(img, b)
                if c is not None:
                    crops.append(c); tags.append(t)
            batch = torch.stack([pre(c) for c in crops])
            with torch.no_grad():
                fi = model.encode_image(batch)
                ft = model.encode_text(tok([q]))
                fi = fi / fi.norm(dim=-1, keepdim=True)
                ft = ft / ft.norm(dim=-1, keepdim=True)
                sims = (fi @ ft.T).squeeze(-1).tolist()
            sd = dict(zip(tags, sims))
            order = sorted(sd.items(), key=lambda x: -x[1])
            margin = order[0][1] - order[1][1] if len(order) > 1 else 0.0
            i_lm = iou(boxes["L2"], boxes["M"])
            i_lg = iou(boxes["L2"], boxes["G"])
            i_mg = iou(boxes["M"], boxes["G"])
            print(f"{k:14s} {margin:7.3f} {sd.get('L2', float('nan')):7.3f} "
                  f"{sd.get('M', float('nan')):7.3f} {sd.get('G', float('nan')):7.3f} "
                  f"{i_lm:7.3f} {i_lg:7.3f} {i_mg:7.3f} {order[0][0]:>5s}  {q[:58]}")
            out[k] = {"stratum": s, "query": q, "sims": sd, "margin": margin,
                      "iou": {"LM": i_lm, "LG": i_lg, "MG": i_mg},
                      "top1": order[0][0], "boxes": boxes,
                      "areas": {t: area(b) for t, b in boxes.items()}}

    # 分层统计
    print("\n" + "=" * 60)
    print(f"{'stratum':8s} {'n':>3s} {'meanMargin':>10s} {'medMargin':>9s} "
          f"{'<0.02':>7s} {'<0.01':>7s}  top1 分布")
    for s in ("easy", "medium", "hard"):
        rs = [v for v in out.values() if v["stratum"] == s]
        if not rs:
            continue
        ms = [r["margin"] for r in rs]
        lt2 = sum(1 for m in ms if m < 0.02) / len(ms)
        lt1 = sum(1 for m in ms if m < 0.01) / len(ms)
        dist = {}
        for r in rs:
            dist[r["top1"]] = dist.get(r["top1"], 0) + 1
        print(f"{s:8s} {len(rs):3d} {np.mean(ms):10.3f} {np.median(ms):9.3f} "
              f"{lt2*100:6.0f}% {lt1*100:6.0f}%  {dist}")

    p = OUT / "sanity30.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {p}")


if __name__ == "__main__":
    main()
