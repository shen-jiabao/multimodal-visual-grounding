# -*- coding: utf-8 -*-
"""非序数抽检图渲染器：Qwen 直接框 vs 当前基线（绿=基线 / 红=Qwen）。

用途：数据到齐后生成分组比对图，供分组核对。

- 「会被哪几种策略改动」**直接调用 `_fuse_qwen.decide()` 得到**，不另写一份判据。
  否则两条链会漂移：原先渲染器用 `IoU<0.5` 判不一致，而 `full` 还额外要求候选
  合法（面积<=0.35），于是 `000212_005`（Qwen 0.6102）出现在图里却不会被改动。
  改为共用 `decide()` 后，"渲染集合 == 策略改动集合"在构造上必然成立。
- 只依赖 Qwen 直接结果 + 基线（不需要 SoM），`qwen3vl_all.json` 齐了即可跑。
- 拼版不是整图缩略，而是**围绕两框并集裁剪后再缩放**：细粒度分歧（D 组）
  框只有整图的千分之几，整图缩略图完全看不出差异，裁剪后才有判读价值。
- 按「要回答的审查问题」分组，避免上千张平铺导致看不完（分组不预判结论）：
    C_BOTH_BIG       两边面积都 > 0.35——都不可信，且不会被任何策略采纳
    C_QWEN_BIG       Qwen 面积 > 0.35——疑 Qwen 合并大框，不会被采纳
    Q_REJECT         Qwen 候选不合法（退化等）——不会被采纳
    B_BASE_BIG       基线面积 > 0.20——疑基线合并宽框（safe/shrink 靶点）
    A_TINY_VS_BIG    基线极小(<1e-3) 且 Qwen 大 3 倍以上——需判断「真目标很小」
                     还是「Qwen 退化成粗框」（探针实例：001038_010 基线 4e-4、
                     Qwen 0.046 扫了整个右下角）
    E_QWEN_NOBOX     Qwen 未给框
    D_SMALL_DISAGREE 其余细粒度分歧

输出（默认 viz_qwen_audit/）：
    groups.json          分组明细（query/IoU/面积/比值/组名），按严重度排序
    groups_index.json    各组条数概览
    sheets/<组>_N.jpg    裁剪拼版（每格 384×384 + 两行说明），便于快速扫
    items/<qid>.png      原分辨率整图（带框），仅每组前 --item 条

用法：
    python _viz_qwen_vs_base.py [--policy any] [--sheet 60] [--item 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402

OUT = R / "viz_qwen_audit"
THUMB = 384
CAP_H = 52
COLS, ROWS = 4, 3
PAD = 0.6          # 裁剪时在两框并集外扩的比例
MIN_CROP = 96      # 裁剪区最小边（像素），避免退化框裁出几个像素
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\consola.ttf",
]
COARSE_RATIO = 3.0
# 基线「极小框」阈值：归一化 1e-3 ≈ 0.032×0.032，1920 图上约 60×35 像素，
# 低于此值再谈「是 Qwen 退化还是目标本来就小」才有意义。
TINY_BASE = 1e-3

GROUP_ORDER = ["C_BOTH_BIG", "C_QWEN_BIG", "Q_REJECT", "B_BASE_BIG",
               "A_TINY_VS_BIG", "E_QWEN_NOBOX", "D_SMALL_DISAGREE"]
GROUP_DESC = {
    "C_BOTH_BIG": "两边都>0.35，都不可信(不采纳)",
    "C_QWEN_BIG": "Qwen 面积>0.35，疑 Qwen 合并大框(不采纳)",
    "Q_REJECT": "Qwen 候选不合法(不采纳)",
    "B_BASE_BIG": "基线面积>0.20，疑基线合并宽框(safe/shrink 靶点)",
    "A_TINY_VS_BIG": "基线极小且 Qwen 大3倍以上，需判断真小目标/Qwen退化",
    "E_QWEN_NOBOX": "Qwen 未输出框",
    "D_SMALL_DISAGREE": "细粒度分歧",
}
# 策略集直接取自 `_fuse_qwen`：两处各写一份必然漂移（曾因此让渲染器把
# 000212_005 当成「full 会改」的条目）。
POLICIES = F.POLICIES

# qwen 策略下「值得重点核对」的分组：只收 Qwen 自身可疑或缺失的条目。
# 不能用「该策略会改动它」来筛——qwen 为准时几乎每条非序数都会变（实测探针
# 33 条里 32 条），那样会产出上万张图，等于没有核对。
QWEN_SUSPECT = {"C_BOTH_BIG", "C_QWEN_BIG", "Q_REJECT",
                "A_TINY_VS_BIG", "E_QWEN_NOBOX"}


def load_font(size: int):
    """加载可用字体，失败则回退 PIL 默认字体。"""
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def fit_text(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> str:
    """按像素宽度截断文本（否则说明会横向溢出、压到相邻格子）。"""
    if d.textlength(text, font=font) <= max_w:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if d.textlength(text[:mid] + "…", font=font) <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


def classify(qb, bb) -> str:
    """按「要回答的审查问题」分组，不做结论预判。

    前三组（C_BOTH_BIG / C_QWEN_BIG / Q_REJECT）的候选不合法，任何策略都不会
    采纳，但它们本身就是 Qwen 的失效指纹，仍有审查价值。
    """
    if qb is None:
        return "E_QWEN_NOBOX"
    aq, ab = F.area(qb), F.area(bb)
    if aq > F.MAX_AREA and ab > F.MAX_AREA:
        return "C_BOTH_BIG"
    if aq > F.MAX_AREA:
        return "C_QWEN_BIG"
    if not F.valid_cand(qb):
        return "Q_REJECT"
    if ab > F.BIG_AREA:
        return "B_BASE_BIG"
    if ab < TINY_BASE and aq >= COARSE_RATIO * ab:
        return "A_TINY_VS_BIG"
    return "D_SMALL_DISAGREE"


def policies_for(bb, qb) -> list:
    """哪些策略会改动这条——直接问 `_fuse_qwen.decide()`，不另立判据。"""
    out = []
    for p in POLICIES:
        if F.decide(p, bb, qb, None, False)[0] is not None:
            out.append(p)
    return out


def region_of(bb, qb, w: int, h: int) -> tuple:
    """取两框并集并外扩，返回正方形像素裁剪区域。"""
    bs = [b for b in (bb, qb) if b]
    x1 = min(b[0] for b in bs) * w
    y1 = min(b[1] for b in bs) * h
    x2 = max(b[2] for b in bs) * w
    y2 = max(b[3] for b in bs) * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, MIN_CROP) * (1 + PAD)
    return (max(0, int(cx - side / 2)), max(0, int(cy - side / 2)),
            min(w, int(cx + side / 2)), min(h, int(cy + side / 2)))


def annotate(im: Image.Image, bb, qb, region=None, size=None) -> Image.Image:
    """裁剪（可选）→ 缩放（可选）→ 画 绿=基线 / 红=Qwen。

    先缩放后画线，线宽不随缩放变细。
    """
    w0, h0 = im.size
    if region:
        x0, y0, x1, y1 = region
        crop = im.crop(region)
    else:
        x0, y0, x1, y1 = 0, 0, w0, h0
        crop = im
    sw, sh = max(1, x1 - x0), max(1, y1 - y0)
    if size:
        crop = crop.resize(size)
    cw, ch = crop.size
    d = ImageDraw.Draw(crop)
    lw = max(2, int(min(cw, ch) * 0.01))
    for b, col in ((bb, (0, 255, 0)), (qb, (255, 0, 0))):
        if not b:
            continue
        d.rectangle([(b[0] * w0 - x0) * cw / sw, (b[1] * h0 - y0) * ch / sh,
                     (b[2] * w0 - x0) * cw / sw, (b[3] * h0 - y0) * ch / sh],
                    outline=col, width=lw)
    return crop


POL_TAG = {"qwen": "qwn", "safe": "saf", "shrink": "shr", "full": "ful"}
# 必须覆盖全部策略：否则新增策略时 caption 会 KeyError（曾漏配 qwen 而崩溃）
assert set(POL_TAG) == set(POLICIES), f"POL_TAG 未覆盖策略: {set(POLICIES) - set(POL_TAG)}"


def caption(row: dict) -> list:
    """生成说明文字两行（未截断，绘制时再按宽度裁剪）。"""
    ratio = row["ratio"]
    pol = "+".join(POL_TAG[p] for p in row["policies"]) or "不采纳"
    return [
        f"{row['qid']}  IoU={row['iou']}  A/B="
        f"{'inf' if ratio is None else f'{ratio:.2f}'}  [{pol}]",
        f"Q={row['qwen_area']} B={row['base_area']}  {row['query'] or ''}",
    ]


def render_sheet(cells: list, path: Path) -> None:
    """把若干 (row, 已标注缩略图) 渲染成一张拼版。"""
    f_cap = load_font(15)
    sheet = Image.new("RGB", (COLS * THUMB, ROWS * (THUMB + CAP_H)),
                      (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (row, img) in enumerate(cells):
        cx, cy = (i % COLS) * THUMB, (i // COLS) * (THUMB + CAP_H)
        for j, line in enumerate(caption(row)):
            d.text((cx + 4, cy + 4 + j * 19),
                   fit_text(d, line, f_cap, THUMB - 8),
                   fill=(0, 0, 0) if j == 0 else (120, 0, 0), font=f_cap)
        sheet.paste(img, (cx, cy + CAP_H))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=88)


def make_row(qid: str, entry: dict, bb, qb, iou=None) -> dict:
    """构造一条审计记录（面积/比值用原始值算，仅展示时量化）。"""
    aq = F.area(qb) if qb else None
    ab = F.area(bb)
    ratio = (aq / ab) if (aq is not None and ab > 0) else None
    pol = policies_for(bb, qb)
    return {
        "qid": qid,
        "query": entry.get("query"),
        "iou": round(iou, 4) if iou is not None else None,
        "base_bbox": bb,
        "qwen_bbox": qb,
        "base_area": round(ab, 5),
        "qwen_area": round(aq, 5) if aq is not None else None,
        "ratio": round(ratio, 3) if ratio is not None else None,
        "group": classify(qb, bb),
        "policies": pol,
        "actionable": bool(pol),
    }


def _open(qid: str, visible: str):
    """按提交里的 visible 路径读图，失败返回 None。"""
    p = F.resolve_img(visible)
    if p is None:
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as ex:
        print(f"  [viz] 读图失败 {qid}: {ex!r}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="any", choices=[*POLICIES, "any"],
                    help="qwen = 只列 Qwen 自身可疑或缺失的条目（与基线无关）；"
                         "safe/shrink/full = 会被该策略改动的条目；"
                         "any = 任一策略会改动（默认，另含 Qwen 失效条目）")
    ap.add_argument("--sheet", type=int, default=60, help="每组拼版最多几条")
    ap.add_argument("--item", type=int, default=12,
                    help="每组另存原分辨率整图的条数（0 表示不存）")
    args = ap.parse_args()

    base = json.loads(F.BASE_JSON.read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    qd = json.loads(F.QD_JSON.read_text(encoding="utf-8"))
    ordinal = set(json.loads(F.CORES_JSON.read_text(encoding="utf-8")))
    print(f"基线 {len(base)} | Qwen直接 {len(qd)} | 序数 {len(ordinal)}")
    if len(qd) != F.N_ENTRIES:
        print(f"[中止] Qwen 直接结果不完整（{len(qd)}/{F.N_ENTRIES}），"
              f"拒绝在部分数据上出图")
        return 2
    assert set(base) == set(test) == set(qd), "键集不一致"

    rows, n_ord = [], 0
    for qid, entry in base.items():
        if qid in ordinal:
            n_ord += 1
            continue
        bb = entry["bbox"]
        qb = qd[qid].get("bbox")
        iou = F.iou(bb, qb) if qb else None
        row = make_row(qid, entry, bb, qb, iou)
        qwen_fail = qb is None or not F.valid_cand(qb)
        if args.policy == "qwen":
            # Qwen 为准时，核对对象是「Qwen 自身可疑或缺失」，与基线无关
            keep = row["group"] in QWEN_SUSPECT
        else:
            # 其余策略：收录会被该策略改动、或 Qwen 失效的条目
            keep = row["actionable"] or qwen_fail
            if keep and args.policy != "any" \
                    and args.policy not in row["policies"]:
                keep = False
        if keep:
            rows.append(row)

    rows.sort(key=lambda r: (GROUP_ORDER.index(r["group"]),
                             -(r["ratio"] if r["ratio"] else 1e9)))
    buckets = {g: [r for r in rows if r["group"] == g] for g in GROUP_ORDER}
    n_non = len(base) - n_ord
    n_act = sum(1 for r in rows if r["actionable"])
    print(f"非序数 {n_non} 条：收录 {len(rows)}"
          f"（可被策略改动 {n_act}，Qwen 失效 {len(rows) - n_act}）"
          f"  策略口径={args.policy}")
    for p in POLICIES:
        print(f"  仅 {p:6s} 会改动 {sum(1 for r in rows if p in r['policies'])}")
    for g in GROUP_ORDER:
        if buckets[g]:
            print(f"  {g:16s} {len(buckets[g]):5d}  {GROUP_DESC[g]}")

    OUT.mkdir(exist_ok=True)
    (OUT / "groups.json").write_text(
        json.dumps({g: buckets[g] for g in GROUP_ORDER},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "groups_index.json").write_text(
        json.dumps({g: {"n": len(buckets[g]), "desc": GROUP_DESC[g]}
                    for g in GROUP_ORDER if buckets[g]},
                   ensure_ascii=False, indent=1), encoding="utf-8")

    per = COLS * ROWS
    for g in GROUP_ORDER:
        items = buckets[g]
        if not items:
            continue
        picked = items[:args.sheet]
        n_sheet = 0
        for si in range(0, len(picked), per):
            cells = []
            for r in picked[si:si + per]:
                im = _open(r["qid"], base[r["qid"]].get("visible"))
                if im is None:
                    continue
                reg = region_of(r["base_bbox"], r["qwen_bbox"], *im.size)
                cells.append((r, annotate(im, r["base_bbox"], r["qwen_bbox"],
                                          reg, (THUMB, THUMB))))
            if cells:
                n_sheet += 1
                render_sheet(cells, OUT / "sheets" / f"{g}_{n_sheet}.jpg")
        print(f"  {g}: 拼版 {n_sheet} 张，覆盖 {min(len(picked), len(items))}"
              f"/{len(items)} 条")

        if args.item:
            f_tag = load_font(26)
            (OUT / "items").mkdir(parents=True, exist_ok=True)
            for r in items[:args.item]:
                im = _open(r["qid"], base[r["qid"]].get("visible"))
                if im is None:
                    continue
                im = annotate(im, r["base_bbox"], r["qwen_bbox"])
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, im.size[0], 62], fill=(255, 255, 255))
                cap = caption(r)
                d.text((8, 4), fit_text(d, cap[0], f_tag, im.size[0] - 16),
                       fill=(0, 0, 0), font=f_tag)
                d.text((8, 32), fit_text(d, cap[1], f_tag, im.size[0] - 16),
                       fill=(120, 0, 0), font=f_tag)
                im.save(OUT / "items" / f"{r['qid']}.png")

    print(f"\n输出目录 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
