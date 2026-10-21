#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLX-Seek 非序数定位驱动。

机制（与官方 `inference.py` 一致，不自造）
    候选框：WeDetect-Base-Uni —— 官方 `load_wedetect_proposals()`，top-100 像素 xyxy
    选择　：`VLXSeekWorker.ground()` → 任务 `grounding_single`
            （模板："Detect a single instance of: {}."）

★能力边界（必须写在最前，避免误读结果）
    模型**只做"选"，不画框**——返回的框必然是输入候选之一。
    ⇒ 输出质量被 **WeDetect 的候选质量**硬性封顶；候选里没有的目标，永远选不出来。
    本条使"全量件"的读数必须解释为「检测器能力 × 选择能力」的乘积。

坐标口径（★易错点）
    官方 `bbox_list` 是**原图像素**坐标（不是 0-1000 / 0-1）。
    我方基线框是 0-1 归一化 ⇒ 喂入前 ×(W,H)，取出后 ÷(W,H)。
    返回坐标的可信性已从源码核实：`predict()` 先 `_order_boxes()` 得到内部排序，
    `_remap_object_tokens()` 把模型答的**内部序号**映射回**调用方序号**，
    `_build_result_bbox_list()` 再用该序号索引 `caller_boxes` ⇒ 返回的就是
    我们喂进去的那组候选框坐标，无歧义。

解码策略（★与官方默认不同，须知悉）
    官方 `predict()` 默认 `temperature=0.2`（`do_sample=True`）⇒ **随机采样**，
    同一输入两次跑结果可能不同。本驱动缺省改为 `--temp 0.0`（贪心、确定性），
    理由：本项目要求读数可复现、且 20 条冒烟闸门不该被采样噪声撞翻。
    若要严格照官方默认，加 `--temp 0.2`。

用法
    python _vlx_direct.py --qids _vlx_smoke20.json --out _vlx_smoke20_out.json --vis
    python _vlx_direct.py --out _vlx_full.json          # 默认全量非序数 4587 条
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

R = Path("/root/rivermind-data")
VLX = R / "vlx-seek"
sys.path.insert(0, str(VLX))

MODEL = str(VLX / "resources" / "VLX-Seek-1.5-10B")
DETECTOR = str(VLX / "resources" / "wedetect_base_uni.pth")
DATA = R / "dataset"
CORES = R / "_ordinal_cores_r2.json"
SAVE_EVERY = 25


def load_qids(a) -> tuple[dict, list[str]]:
    """返回 (queries 全表, 本次要跑的键序)。"""
    q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads(CORES.read_text(encoding="utf-8")))
    if a.qids:
        sel = json.loads(Path(a.qids).read_text(encoding="utf-8"))
    else:
        sel = [k for k in q if k not in cores]
    sel = [k for k in sel if k in q]
    if a.limit:
        sel = sel[: a.limit]
    return q, sel


def save(out_path: Path, res: dict) -> None:
    """原子落盘：先写 .tmp 再 os.replace，避免半截文件被当成完整产物。"""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qids", default=None, help="条目清单 JSON；缺省=全部非序数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vis", action="store_true", help="导出可视化图供目检")
    ap.add_argument("--vis-dir", default=str(R / "vlx_vis"))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="0=贪心（缺省，确定性）；官方默认 0.2（随机采样）")
    a = ap.parse_args()

    q, sel = load_qids(a)
    out_path = Path(a.out)
    print(f"条目 {len(sel)} 条 | 输出 {out_path}", flush=True)

    # ---- 断点续跑：已完成的直接跳过 ----
    res: dict = {}
    if out_path.exists():
        try:
            res = json.loads(out_path.read_text(encoding="utf-8"))
            print(f"  续跑：已有 {len(res)} 条", flush=True)
        except Exception:
            print("  既有产物解析失败，从零开始", flush=True)
            res = {}

    from PIL import Image  # noqa: E402
    from inference import load_wedetect_proposals  # noqa: E402
    from vlx_seek_worker import VLXSeekWorker  # noqa: E402

    t0 = time.time()
    worker = VLXSeekWorker(MODEL, device="cuda")
    print(f"  模型已加载（{time.time()-t0:.0f}s）", flush=True)

    prop_cache: dict[str, list[list[float]]] = {}   # 按图缓存候选（图 ≪ 条目）
    vis_dir = Path(a.vis_dir)
    if a.vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    todo = [k for k in sel if k not in res]
    print(f"  待跑 {len(todo)} 条", flush=True)

    t0 = time.time()
    t_det = 0.0        # 候选生成累计（官方函数每图重建检测器，是主要固定开销）
    t_mod = 0.0        # VLX-Seek 选择累计
    for i, k in enumerate(todo, 1):
        v = q[k]
        name = os.path.basename(v["visible"])
        rec: dict = {"query": v["query"], "img": name}
        try:
            img = Image.open(DATA / "Images" / "visible" / name).convert("RGB")
            W, H = img.size

            if name not in prop_cache:
                _t = time.time()
                prop_cache[name] = load_wedetect_proposals(img, DETECTOR)
                t_det += time.time() - _t
            boxes = prop_cache[name]
            rec["n_cand"] = len(boxes)

            _t = time.time()
            out = worker.ground(img, boxes, v["query"],
                                max_new_tokens=a.max_new_tokens,
                                temperature=a.temp)
            t_mod += time.time() - _t
            rec["answer"] = str(out.get("answer", ""))[:300]
            rb = out.get("result_bbox_list") or []
            if rb:
                b0 = rb[0]
                box = [float(b0["xmin"]) / W, float(b0["ymin"]) / H,
                       float(b0["xmax"]) / W, float(b0["ymax"]) / H]
                # 归一并夹到 [0,1]（像素框经裁剪保证有序，除不尽时仍可能越界）
                box = [max(0.0, min(1.0, x)) for x in box]
                rec["bbox"] = [round(x, 6) for x in box]
                rec["object_index"] = b0.get("object_index")
                rec["label"] = b0.get("label")
                rec["n_ret"] = len(rb)
            else:
                rec["bbox"] = None      # 模型未选中任何候选 → 装配时保留基线
            if a.vis:
                from PIL import ImageDraw
                im = img.copy()
                d = ImageDraw.Draw(im)
                for bb in boxes:
                    d.rectangle(bb, outline=(0, 128, 255), width=2)
                if rec.get("bbox"):
                    x1, y1, x2, y2 = (rec["bbox"][0] * W, rec["bbox"][1] * H,
                                      rec["bbox"][2] * W, rec["bbox"][3] * H)
                    d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
                im.save(vis_dir / f"{k}.jpg", quality=88)
        except Exception as ex:
            rec["err"] = f"{type(ex).__name__}: {ex}"[:200]
            rec["bbox"] = None
        res[k] = rec

        if i % 10 == 0 or i == len(todo):
            dt = time.time() - t0
            rate = i / dt if dt else 0
            print(f"  [{i}/{len(todo)}] {rate:.2f} 条/s | "
                  f"err {sum(1 for x in res.values() if x.get('err'))} | "
                  f"无框 {sum(1 for x in res.values() if not x.get('bbox'))} | "
                  f"累计 检测 {t_det:.0f}s / 选择 {t_mod:.0f}s", flush=True)
        if i % SAVE_EVERY == 0:
            save(out_path, res)

    save(out_path, res)
    nb = sum(1 for v in res.values() if v.get("bbox"))
    ne = sum(1 for v in res.values() if v.get("err"))
    print(f"\n完成：{len(res)} 条 | 有框 {nb} | 无框 {len(res)-nb} | err {ne}", flush=True)
    print(f"已写 {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
