# -*- coding: utf-8 -*-
"""抽查可视化：读 result_gd_3modal_v2.json 已算出的条目，按场景挑几张，
把该场景下每个 query 的预测框画在可见光图上并附文字，输出到 viz_v2_check/。"""
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

RESULT = Path(r"d:\Desktop\2026AiC\result_gd_3modal_v2.json")
DATA_ROOT = Path(r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理")
OUT_DIR = Path(r"d:\Desktop\2026AiC\viz_v2_check")
COPY = Path(r"d:\Desktop\2026AiC\_tmp_result_copy.json")

# 颜色/空间关键词（用于优先挑选有难度的场景）
INTEREST = ["left", "right", "behind", "front", "far", "near", "background",
            "foreground", "closest", "farthest", "above", "below",
            "yellow", "green", "red", "pink", "blue", "white", "black",
            "brown", "orange", "purple", "gray", "grey", "person",
            "driving", "empty", "child", "sign"]

PALETTE = [(0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
           (255, 0, 0), (255, 0, 255), (180, 105, 255), (0, 215, 255)]


def imread_u(p):
    raw = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def wrap(text, width):
    """按像素宽度粗略换行（英文按字符）。"""
    if not text:
        return [""]
    lines, cur = [], ""
    for ch in text:
        cur += ch
        # 每 78 字符约 1 行（8pt 字号），按行数拆
        if len(cur) >= width:
            lines.append(cur)
            cur = ""
    if cur:
        lines.append(cur)
    return lines


def main():
    # 复制一份再解析，避免与后台任务的写入冲突
    shutil.copyfile(RESULT, COPY)
    with open(COPY, "r", encoding="utf-8") as f:
        data = json.load(f)
    os.remove(COPY)
    print(f"已读 {len(data)} 条（当前 checkpoint 进度内的结果）")

    # 按场景分组
    scenes = {}
    for qid, rec in data.items():
        if "bbox" not in rec:
            continue
        scene = qid.split("_")[0]
        scenes.setdefault(scene, []).append((qid, rec))
    print(f"共 {len(scenes)} 个场景")

    # 挑选：含关键词 query 数多的场景优先
    def score(items):
        return sum(1 for _, r in items if any(k in r["query"].lower()
                                              for k in INTEREST))

    ranked = sorted(scenes.items(), key=lambda kv: (len(kv[1]) >= 2, score(kv[1]),
                                                    len(kv[1])), reverse=True)
    chosen = [s for s, _ in ranked[:6]]
    print(f"抽查场景: {chosen}")

    OUT_DIR.mkdir(exist_ok=True)
    for scene in chosen:
        items = sorted(scenes[scene], key=lambda t: t[0])
        rec0 = items[0][1]
        img = imread_u(DATA_ROOT / rec0["visible"])
        if img is None:
            print(f"[跳过] {scene} 图像读取失败")
            continue
        H, W = img.shape[:2]
        disp_w = 760
        disp_h = int(H * disp_w / W)
        cells = []
        for idx, (qid, rec) in enumerate(items):
            cell = cv2.resize(img, (disp_w, disp_h))
            box = rec.get("bbox")
            if box and len(box) == 4:
                x1, y1, x2, y2 = [float(v) for v in box]
                color = PALETTE[idx % len(PALETTE)]
                p1 = (int(x1 * disp_w), int(y1 * disp_h))
                p2 = (int(x2 * disp_w), int(y2 * disp_h))
                cv2.rectangle(cell, p1, p2, color, 3)
                tag = f"{idx}:{qid}"
                cv2.putText(cell, tag, (max(0, p1[0]), max(16, p1[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                            cv2.LINE_AA)
            # 下方文字条（query 全文）
            cap_h = 56
            cap = np.full((cap_h, disp_w, 3), 245, np.uint8)
            query = rec.get("query", "")
            lines = wrap(f"[{idx}] {qid}: {query}", 110)
            for li, ln in enumerate(lines[:2]):
                cv2.putText(cap, ln, (6, 18 + li * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1,
                            cv2.LINE_AA)
            cells.append(np.vstack([cell, cap]))
        panel = np.vstack(cells)
        out_p = OUT_DIR / f"{scene}_v2.png"
        cv2.imencode(".png", panel)[1].tofile(str(out_p))
        print(f"已保存 {out_p}  (含 {len(items)} 个 query)")
    print("done")


if __name__ == "__main__":
    main()
