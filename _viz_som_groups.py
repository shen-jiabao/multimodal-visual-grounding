# -*- coding: utf-8 -*-
"""序数 SoM 组级复核图：画出该组全部候选 + 选中框 + 基线。

背景：`n_cands < N` 的「欠枚举」组——组内问句声明了 N 个实例，但枚举只
给出 n_cands 个候选。需判断这些组的 SoM 结果是否选错了实例。

关键前提（已实测核实）：som 记录存有 `bbox_all`（全部候选坐标），
1641/2014 条含该字段，因此可直接画出候选编号总览；无需靠"组内选中框
是否互相重合"来间接推断（那是只有单个框时才不得已的办法）。

判读方式（每格一组）：
    橙色细框 + 编号  = 候选表全部候选（1 = 从左到右第 1 个）
    红色粗框        = 该条的选中框
    绿色细框        = 当前锁定提交（基线）
    文字 k1→2 k2→3  = 组内各问句（序号 k）最终选中了第几个候选

判据：
- 不同 k 落到同一编号 → 这些条目里最多只有一条可能对（错位）
- n_cands < N 时，必有 (N - n_cands) 个 k 在候选表里找不到对应实例
  （这部分已由 `_fuse_qwen.SOM_UNDERCOVER` 在 k > n_cands 时拦截）

复用：`_fuse_qwen.resolve_img`、`_viz_qwen_vs_base.load_font/fit_text`。

用法：
    python _viz_som_groups.py                 # 欠枚举且已被采纳的组
    python _viz_som_groups.py --mode all      # 全部序数组
    python _viz_som_groups.py --qids 002144_001 002620_001
输出 viz_out/som_groups/：sheets/*.jpg + groups.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _viz_qwen_vs_base as V  # noqa: E402
from _analyze_som_ordinal import direction  # noqa: E402

OUT = R / "viz_out" / "som_groups"
THUMB = 420
CAP_H = 56
COLS, ROWS = 4, 3
PAD = 0.25          # 裁剪时在并集外扩的比例
MIN_CROP = 140      # 裁剪区最小边（像素）
C_ORANGE = (255, 140, 0)
C_RED = (230, 0, 0)
C_GREEN = (0, 200, 0)


def union_region(boxes: list, w: int, h: int) -> tuple:
    """取所有框并集并外扩，返回正方形像素裁剪区域。"""
    bs = [b for b in boxes if b]
    if not bs:
        return 0, 0, w, h
    x1 = min(b[0] for b in bs) * w
    y1 = min(b[1] for b in bs) * h
    x2 = max(b[2] for b in bs) * w
    y2 = max(b[3] for b in bs) * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, MIN_CROP) * (1 + PAD)
    return (max(0, int(cx - side / 2)), max(0, int(cy - side / 2)),
            min(w, int(cx + side / 2)), min(h, int(cy + side / 2)))


def draw_group(im: Image.Image, cands: list, sel: list, base,
               region: tuple, size: tuple) -> Image.Image:
    """裁剪 → 缩放 → 画候选(橙+编号) / 选中(红粗) / 基线(绿)。

    先缩放后画线，避免线宽随缩放变细。
    """
    w0, h0 = im.size
    x0, y0, x1, y1 = region
    crop = im.crop(region).resize(size)
    cw, ch = crop.size
    sw, sh = max(1, x1 - x0), max(1, y1 - y0)
    d = ImageDraw.Draw(crop)
    f_num = V.load_font(max(14, int(min(cw, ch) * 0.05)))
    lw = max(2, int(min(cw, ch) * 0.008))

    def xy(b):
        """归一化框 -> 裁剪后的像素框。"""
        return [(b[0] * w0 - x0) * cw / sw, (b[1] * h0 - y0) * ch / sh,
                (b[2] * w0 - x0) * cw / sw, (b[3] * h0 - y0) * ch / sh]

    if base:
        d.rectangle(xy(base), outline=C_GREEN, width=lw)
    for i, b in enumerate(cands, 1):
        p = xy(b)
        d.rectangle(p, outline=C_ORANGE, width=lw)
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        d.text((cx - 6, cy - 8), str(i), fill=C_ORANGE, font=f_num)
    if sel:
        d.rectangle(xy(sel), outline=C_RED, width=lw * 2)
    return crop


def caption(row: dict) -> list:
    """两行说明：组信息 + 各问句的 序号->选中编号 映射。"""
    ks = " ".join(f"k{k}→{n}" for k, n in row["pairs"])
    return [
        f"{row['img']}  {row['core'][:30]!r}  N={row['N']} "
        f"n_cands={row['n_cands']}",
        f"{ks}    绿=基线 橙=候选 红=选中",
    ]


def render_sheet(cells: list, path: Path, cap_fn=None) -> None:
    """把若干 (row, 已绘制图) 渲染成拼版。cap_fn 可覆盖说明文字生成。"""
    f_cap = V.load_font(15)
    sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + CAP_H)),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (row, img) in enumerate(cells):
        cx, cy = (i % COLS) * THUMB, (i // COLS) * (THUMB + CAP_H)
        for j, line in enumerate((cap_fn or caption)(row)):
            d.text((cx + 4, cy + 4 + j * 20),
                   V.fit_text(d, line, f_cap, THUMB - 8),
                   fill=(0, 0, 0) if j == 0 else (150, 60, 0), font=f_cap)
        sheet.paste(img, (cx, cy + CAP_H))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def build_det_rows(som: dict, cores: dict, test: dict, base: dict,
                   only: set) -> list:
    """构造「确定性选择 vs 模型选择 vs 基线」三方对比记录。

    只收录 `expected_ordinal_num` 可算、且模型答案与期望不符的可疑条目——
    这些正是 det 会修正、det_rb 会回退的靶心。
    """
    rows = []
    untrusted = F.untrusted_ordinals()
    for qid in sorted(cores):
        if qid in F.FROZEN or qid in untrusted:
            continue
        if only and qid not in only:
            continue
        rec = som.get(qid) or {}
        cands = rec.get("bbox_all") or []
        q = test[qid]["query"]
        exp = F.expected_ordinal_num(q, cands, rec.get("n_cands"))
        if exp is None or rec.get("num") == exp:
            continue
        rows.append({
            "qid": qid, "img": Path(test[qid]["visible"]).name,
            "core": rec.get("core") or "", "k": F.ordinal_index(q),
            "d": F.direction(q), "nc": len(cands), "exp": exp,
            "model_num": rec.get("num"), "query": q,
            "cands": cands, "model_box": rec.get("bbox"),
            "det_box": cands[exp - 1], "base_box": base[qid]["bbox"],
        })
    rows.sort(key=lambda r: (r["img"], r["core"]))
    return rows


def draw_det(im: Image.Image, row: dict, region: tuple, size: tuple):
    """裁剪→缩放→画 绿=基线 / 橙=模型选的 / 红=det 选的 / 灰=其余候选。"""
    w0, h0 = im.size
    x0, y0, x1, y1 = region
    crop = im.crop(region).resize(size)
    cw, ch = crop.size
    sw, sh = max(1, x1 - x0), max(1, y1 - y0)
    d = ImageDraw.Draw(crop)

    def xy(b):
        return [(b[0] * w0 - x0) * cw / sw, (b[1] * h0 - y0) * ch / sh,
                (b[2] * w0 - x0) * cw / sw, (b[3] * h0 - y0) * ch / sh]

    lw = max(2, int(min(cw, ch) * 0.008))
    for b in row["cands"]:
        d.rectangle(xy(b), outline=(170, 170, 170), width=1)
    if row["base_box"]:
        d.rectangle(xy(row["base_box"]), outline=C_GREEN, width=lw)
    if row["model_box"]:
        d.rectangle(xy(row["model_box"]), outline=C_ORANGE, width=lw)
    d.rectangle(xy(row["det_box"]), outline=C_RED, width=lw * 2)
    return crop


def caption_det(row: dict) -> list:
    """说明：方向/序号/候选数/模型编号/期望编号。"""
    return [
        f"{row['qid']} {row['core'][:22]!r} {row['d']} k={row['k']} "
        f"nc={row['nc']}",
        f"模型答{row['model_num']} 期望{row['exp']}   绿基线 橙模型 红det",
    ]


def build_rows(som: dict, cores: dict, test: dict, base: dict,
               mode: str, only: set) -> list:
    """按 (图, 核心词) 分组，构造复核记录。"""
    grp = defaultdict(list)
    for qid, c in cores.items():
        img = Path(test[qid]["visible"]).name if qid in test else "?"
        grp[(img, c.lower())].append(qid)

    rows = []
    for (img, core), qids in grp.items():
        qids = sorted(qids)
        rec0 = som.get(qids[0]) or {}
        nc = rec0.get("n_cands") or 0
        rs = rec0.get("reason", "ok")
        cands = rec0.get("bbox_all") or []
        N = len(qids)
        pairs, sels, bases = [], [], []
        for q in qids:
            rec = som.get(q) or {}
            k = F.ordinal_index(test[q]["query"]) if q in test else None
            num = rec.get("num")
            if rec.get("bbox"):
                pairs.append((k, num))
                sels.append(rec["bbox"])
            bases.append(base[q]["bbox"])
        if not pairs:
            continue
        if mode == "under" and not (nc < N and rs == "ok"):
            continue
        if only and not (set(qids) & only):
            continue
        rows.append({"img": img, "core": core, "N": N, "n_cands": nc,
                     "reason": rs, "qids": qids, "pairs": pairs,
                     "cands": cands, "sel": sels[0],
                     "base": bases[0], "bases": bases})
    rows.sort(key=lambda r: (r["n_cands"] - r["N"], -r["N"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="under",
                    choices=["under", "all", "det"],
                    help="under=欠枚举且已采纳的组；all=全部序数组；"
                         "det=确定性选择 vs 模型选择 vs 基线（三方对比）")
    ap.add_argument("--qids", nargs="*", default=[], help="只渲染含这些 qid 的组")
    ap.add_argument("--sheet", type=int, default=48, help="最多渲染多少组")
    args = ap.parse_args()

    som = json.loads((R / "qwen3vl_som_all.json").read_text(encoding="utf-8"))
    cores = json.loads(F.CORES_JSON.read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    base = json.loads(F.BASE_JSON.read_text(encoding="utf-8"))

    rows = build_rows(som, cores, test, base, args.mode, set(args.qids))

    if args.mode == "det":
        # 三方对比：确定性选择 vs 模型选择 vs 基线
        drows = build_det_rows(som, cores, test, base, set(args.qids))
        print(f"模式=det  可疑条目={len(drows)}"
              f"（模型答案与期望编号不符，即 det 修正 / det_rb 回退的靶心）")
        from collections import Counter as _C
        print("  方向分布:", dict(_C(r["d"] for r in drows)))
        print("  按核心词 top8:",
              [f"{c}:{n}" for c, n in _C(r["core"] for r in drows).most_common(8)])
        bare = [r for r in drows if F.det_ordinal("det", {"bbox_all": r["cands"],
                                                          "n_cands": r["nc"]},
                                                  r["query"])[1] == "ORD_DET_INVALID"]
        print(f"  其中 det 无法采纳（期望候选非法）: {len(bare)}")

        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "det_rows.json").write_text(
            json.dumps([{k: v for k, v in r.items()
                         if k not in ("cands", "model_box", "det_box",
                                      "base_box")}
                        for r in drows], ensure_ascii=False, indent=1),
            encoding="utf-8")
        picked = drows[:args.sheet]
        per = COLS * ROWS
        n_sheet = 0
        for i in range(0, len(picked), per):
            cells = []
            for r in picked[i:i + per]:
                p = F.resolve_img(test[r["qid"]]["visible"])
                if p is None:
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                except Exception as ex:
                    print(f"  [skip] 读图失败 {r['qid']}: {ex!r}")
                    continue
                boxes = list(r["cands"]) + [r["base_box"], r["det_box"]]
                if r["model_box"]:
                    boxes.append(r["model_box"])
                reg = union_region(boxes, *im.size)
                cells.append((r, draw_det(im, r, reg, (THUMB, THUMB))))
            if cells:
                n_sheet += 1
                render_sheet(cells, OUT / "sheets" / f"det_{n_sheet}.jpg",
                             cap_fn=caption_det)
        print(f"  拼版 {n_sheet} 张，覆盖 {min(len(picked), len(drows))}"
              f"/{len(drows)} 条")
        print(f"输出 {OUT}")
        return 0

    n_items = sum(len(r["qids"]) for r in rows)
    print(f"模式={args.mode}  组数={len(rows)}  覆盖条目={n_items}")
    no_cand = sum(1 for r in rows if not r["cands"])
    print(f"  其中缺 bbox_all 的组: {no_cand}")
    same = [r for r in rows
            if len({n for _, n in r["pairs"]}) < len(r["pairs"])]
    print(f"  组内出现重复选中编号的组: {len(same)}"
          f"（覆盖 {sum(len(r['qids']) for r in same)} 条）")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "groups.json").write_text(
        json.dumps([{k: v for k, v in r.items()
                     if k not in ("cands", "sel", "base", "bases")}
                    for r in rows], ensure_ascii=False, indent=1),
        encoding="utf-8")

    picked = rows[:args.sheet]
    per = COLS * ROWS
    n_sheet = 0
    for i in range(0, len(picked), per):
        cells = []
        for r in picked[i:i + per]:
            p = F.resolve_img(test[r["qids"][0]]["visible"]) \
                if r["qids"][0] in test else None
            if p is None:
                continue
            try:
                im = Image.open(p).convert("RGB")
            except Exception as ex:
                print(f"  [skip] 读图失败 {r['img']}: {ex!r}")
                continue
            boxes = list(r["cands"]) + list(r["bases"])
            if r["sel"]:
                boxes.append(r["sel"])
            reg = union_region(boxes, *im.size)
            cells.append((r, draw_group(im, r["cands"], r["sel"], r["base"],
                                        reg, (THUMB, THUMB))))
        if cells:
            n_sheet += 1
            render_sheet(cells, OUT / "sheets" /
                         f"som_{args.mode}_{n_sheet}.jpg")
    print(f"拼版 {n_sheet} 张，覆盖 {min(len(picked), len(rows))}/{len(rows)} 组")
    print(f"输出 {OUT}")

    # 明细打印：重复编号的组（最可疑）
    print("\n组内重复选中编号 top15（不同 k 落到同一候选 = 错位）：")
    for r in same[:15]:
        ks = " ".join(f"k{k}→{n}" for k, n in r["pairs"])
        print(f"  {r['img']} {r['core'][:26]!r:28s} N={r['N']} "
              f"nc={r['n_cands']}  {ks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())