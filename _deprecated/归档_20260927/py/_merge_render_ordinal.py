# -*- coding: utf-8 -*-
"""序数层批量结果：合并探针池 + 批量枚举，按 axis/rank 取第 N 并渲染验证图。

输入：
  locate_multi_probe_000023.json   （图 000023 的 P3 池，5 个石墩）
  locate_ordinal_batch.json        （7 张新图的枚举结果，远端下载）
  variants_30.json / test.json     （query 文案、图片路径）
输出：
  ordinal_select_result.json       （每问：选中框、候选数、截断标志）
  viz_out/{qid}_ordinal_pool.png   （绿编号全候选 + 红选中）

TRUNCATED 判定（不静默兜底）：rank > 去重后候选数 时标记 truncated=true，
渲染图选中框改为黄色并标注 TRUNC，交人工判读。
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from locate_ordinal_select import select_nth

DATA_ROOT = r"d:\Desktop\2026AiC\初赛数据集-基于大模型的多模态视觉理解与推理"
PROBE_000023 = r"d:\Desktop\2026AiC\locate_multi_probe_000023.json"
BATCH_JSON = r"d:\Desktop\2026AiC\locate_ordinal_batch.json"
RETRY_JSON = r"d:\Desktop\2026AiC\locate_ordinal_retry.json"
VARIANTS = r"d:\Desktop\2026AiC\variants_30.json"
OUT_JSON = r"d:\Desktop\2026AiC\ordinal_select_result.json"
OUT_DIR = r"d:\Desktop\2026AiC\viz_out"

# 项目规则：面积超过整图 35% 的框视为合并/背景框，排除（如 001773 退化全幅框）
MAX_AREA_RATIO = 0.35
# 类别污染例：016333 二次批只枚举白鹭鸶（排除黑天鹅），不再并入首轮候选
RETRY_ONLY_IMAGES = {"016333"}

GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)


def imread_unicode(path: str) -> np.ndarray:
    """兼容中文路径读图。"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def imwrite_unicode(path: str, img: np.ndarray) -> None:
    """兼容中文路径写图。"""
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise OSError(path)
    buf.tofile(path)


def to_px(box, w, h):
    """归一化 xyxy -> 像素并夹边。"""
    x1, y1, x2, y2 = box
    return (max(0, int(round(x1 * w))), max(0, int(round(y1 * h))),
            min(w - 1, int(round(x2 * w))), min(h - 1, int(round(y2 * h))))


def tag_box(img, box_px, color, tag, thickness):
    """画框 + 角标。"""
    x1, y1, x2, y2 = box_px
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if tag:
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)),
                      (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, tag, (x1 + 4, max(th, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)


def axis_order(axis: str):
    """槽位 axis -> (排序轴, 顺序)。"""
    mapping = {
        "left_to_right": ("x", "asc"),
        "right_to_left": ("x", "desc"),
        "top_to_bottom": ("y", "asc"),
        "bottom_to_top": ("y", "desc"),
    }
    return mapping.get(axis, ("x", "asc"))


def load_pools(variants):
    """汇总 {图id: {boxes, visible, core}}。

    来源合并（按"图片 id"聚合，而非枚举 tag）：
      - 000023        ：用 P3 探针池。
      - 首轮批量 + 二次批：同一张图聚合所有变体的候选，二次批优先；
                        RETRY_ONLY_IMAGES 的图不再并入首轮候选（排除类别污染）。
      候选先过滤 area>MAX_AREA_RATIO 的合并/背景框。
    """
    pools = {}

    def img_of(visible: str) -> str:
        """由可见图路径取图片 id（如 010033a/010033b -> 010033 已在 load 端指定）。"""
        return os.path.splitext(os.path.basename(visible))[0]

    def filtered(boxes, name=""):
        """排除面积超阈值及退化框，返回归一化列表。"""
        out = []
        for b in boxes:
            x1, y1, x2, y2 = b
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > MAX_AREA_RATIO:
                continue
            out.append(b)
        return out

    # 000023 探针（P3）
    probe = json.load(open(PROBE_000023, encoding="utf-8"))
    pools["000023"] = {"boxes": filtered(probe["P3_count_then_box"]["boxes"],
                                         "000023"),
                       "visible": "Images/visible/000023.png",
                       "core": "stone pier"}

    first = json.load(open(BATCH_JSON, encoding="utf-8")) \
        if os.path.exists(BATCH_JSON) else {}
    retry = json.load(open(RETRY_JSON, encoding="utf-8")) \
        if os.path.exists(RETRY_JSON) else {}

    # 以图 id 为键聚合候选框 + 可见路径
    agg = {}

    def add(img_id, boxes, visible, core):
        if img_id not in agg:
            agg[img_id] = {"boxes": [], "visible": visible, "core": core}
        agg[img_id]["boxes"].extend(boxes)

    for img_id, info in first.items():
        if "boxes" in info:
            add(img_id, info["boxes"], info["visible"], info.get("core", ""))
    for tag, info in retry.items():
        # 按可见图路径取图片 id 聚合（tag 疑似带 a/b 后缀，不能直接用）
        img_id = img_of(info.get("visible", ""))
        if "boxes" in info:
            add(img_id, info["boxes"], info["visible"], info.get("core", ""))

    for img_id, info in agg.items():
        raw_boxes = info["boxes"]
        if img_id in RETRY_ONLY_IMAGES:
            # 仅保留该图的所有二次批变体候选，丢弃首轮（排除污染类别）
            raw_boxes = []
            for tag, rinfo in retry.items():
                if img_of(rinfo.get("visible", "")) == img_id:
                    raw_boxes.extend(rinfo.get("boxes", []))
        pools[img_id] = {"boxes": filtered(raw_boxes, img_id),
                         "visible": info["visible"],
                         "core": info.get("core", "")}
    return pools


def main():
    """逐序数问取第 N、判定截断、渲染。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    variants = json.load(open(VARIANTS, encoding="utf-8"))
    test_data = json.load(
        open(os.path.join(DATA_ROOT, "test.json"), encoding="utf-8"))
    pools = load_pools(variants)

    ordinal_qids = [q for q, d in variants.items()
                    if d["meta"].get("ordinal") is not None]
    results = {}
    for qid in sorted(ordinal_qids):
        meta = variants[qid]["meta"]
        rank = meta["ordinal"]
        axis, order = axis_order(meta.get("axis"))
        visible = test_data[qid]["visible"]
        img_id = os.path.splitext(os.path.basename(visible))[0]

        if img_id not in pools:
            results[qid] = {"status": "missing_pool", "img": img_id}
            print(f"[MISS] {qid} 图 {img_id} 枚举结果缺失")
            continue
        raw_boxes = pools[img_id]["boxes"]
        # 先用 rank=1 拿完整去重排序池，再判断截断
        _, pool = select_nth(raw_boxes, 1, axis=axis, order=order)
        truncated = abs(rank) > len(pool)
        sel, _ = select_nth(raw_boxes, rank, axis=axis, order=order)

        if not pool:
            # 空池（模型对该类别吐 None）：记录并渲染标注图，不画选中框
            results[qid] = {
                "img": img_id, "rank": rank, "axis": meta.get("axis"),
                "n_pool": 0, "truncated": True, "no_candidate": True,
                "bbox": None, "pool": [],
            }
            print(f"[NONE] {qid} rank={rank} 模型未枚举到任何候选")
            canvas = imread_unicode(os.path.join(DATA_ROOT, visible))
            h, w = canvas.shape[:2]
            banner = np.full((56, w, 3), 30, dtype=np.uint8)
            cv2.putText(banner, f"{qid}  {test_data[qid]['query']}", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
            cv2.putText(banner, "NO CANDIDATE FROM ENUMERATION", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2, cv2.LINE_AA)
            out = np.vstack([banner, canvas])
            imwrite_unicode(os.path.join(OUT_DIR, f"{qid}_ordinal_pool.png"), out)
            continue

        results[qid] = {
            "img": img_id, "rank": rank, "axis": meta.get("axis"),
            "n_pool": len(pool), "truncated": truncated,
            "bbox": [round(float(v), 6) for v in sel],
            "pool": [[round(float(v), 6) for v in b] for b in pool],
        }
        flag = "TRUNC" if truncated else "ok"
        print(f"[{flag}] {qid} rank={rank} pool={len(pool)} "
              f"bbox={[round(v,3) for v in sel]}")

        # 渲染
        canvas = imread_unicode(os.path.join(DATA_ROOT, visible))
        h, w = canvas.shape[:2]
        for i, b in enumerate(pool, start=1):
            tag_box(canvas, to_px(b, w, h), GREEN, str(i), 2)
        pick_color = YELLOW if truncated else RED
        pick_tag = f"TRUNC #{rank}/{len(pool)}" if truncated else f"pick #{rank}"
        tag_box(canvas, to_px(sel, w, h), pick_color, pick_tag, 5)
        banner = np.full((56, w, 3), 30, dtype=np.uint8)
        cv2.putText(banner, f"{qid}  {test_data[qid]['query']}", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        out = np.vstack([banner, canvas])
        imwrite_unicode(os.path.join(OUT_DIR, f"{qid}_ordinal_pool.png"), out)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n结果 {OUT_JSON}；图 {OUT_DIR}")


if __name__ == "__main__":
    main()