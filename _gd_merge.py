# -*- coding: utf-8 -*-
"""GD 候选表的本地取舍、跨枚举器召回对照与变体生成（判据复用 `_fuse_qwen`）。

输入 `_gd_enum_raw.json`：`visible||core -> {bbox_all, scores}`（只采集，未去重未排序）。
阈值/去重策略全在这里选，改多少次都不必再占算力。

与 `_enum_merge.py` 的关键区别
- 那份是 8B 的多种提示**合并**（已线上实测否掉，−24.8%）；这份是**单一检测器**的候选池，
  按分数阈值 + NMS 去重，机制上更接近"一份完整的左到右候选表"。

**验证口径（吸取上一轮教训）**
- 上一轮我用「控制组保真度」判断策略，结果**用错了总体**：控制组是"模型数得清"的样本，
  目标组是"模型数不清"的样本，两者不同机制，低扰动不能外推。
- 本轮控制组只做**跨枚举器召回对照**（两个不同机制的枚举器在"8B 数得清"的地方是否一致），
  这是**必要非充分**条件——它能剔除明显坏的表，但**不能证明**目标组上正确。
  目标组的最终判据只能是线上分数或人工目检。

用法
    python -B _gd_merge.py --sweep            # 阈值扫描 + 控制组召回对照
    python -B _gd_merge.py --target --render 10
    python -B _gd_merge.py --zip --floor 0.20 --nms 0.3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))
import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
SUB = R / "submission_zips" / "semifinal"
GD_RAW = R / "_gd_enum_raw.json"
TARGETS = R / "_enum_targets.json"
CORES = R / "_ordinal_cores_r2.json"
VIZ = R / "viz_gd_enum"
AXES = ("x", "y")
FLOORS = (0.15, 0.20, 0.25, 0.30, 0.35)
NMS_IOUS = (0.3, 0.5)
MAX_N = 40


def table(rec: dict, floor: float, nms_iou: float) -> list:
    """按分数阈值 + NMS 去重 + 按 x 中心排序，得到候选表。

    NMS 按分数降序贪心：与已保留框 IoU 超过阈值者丢弃——同 `predict_gd_3modal` 主
    管线口径（原为 0.3），避免同一实例的重复框把"第 k 个"的索引顶偏。
    """
    pairs = [(s, b) for s, b in zip(rec.get("scores") or [], rec.get("bbox_all") or [])
             if s >= floor and F.ordered(b)]
    pairs.sort(key=lambda t: -t[0])
    keep: list = []
    for _s, b in pairs:
        if any(F.iou(b, o) > nms_iou for o in keep):
            continue
        keep.append(list(b))
        if len(keep) >= MAX_N:
            break
    return sorted(keep, key=lambda b: (b[0] + b[2]) / 2)


def pick(query: str, tab: list):
    """文本推算取第 k 项；前提不满足返回 None（判据全部来自 `_fuse_qwen`）。"""
    exp = F.expected_ordinal_num(query, tab, None, AXES)
    if exp is None:
        return None
    seq = F.ordered_candidates(query, tab, AXES)
    if exp > len(seq):
        return None
    nb = F.norm(seq[exp - 1])
    return nb if F.valid_cand(nb) else None


def load():
    raw = json.loads(GD_RAW.read_text(encoding="utf-8"))
    tg = json.loads(TARGETS.read_text(encoding="utf-8"))
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = json.loads(CORES.read_text(encoding="utf-8"))
    return raw, tg, q, cores


def sweep(raw, tg, q, cores, base) -> None:
    """阈值扫描：控制组看**跨枚举器召回**（GD 表是否含 8B 所选那一个实例）。"""
    print("===== 1) 阈值扫描（控制组：跨枚举器召回对照，必要非充分）=====")
    print(f"{'floor':>6s} {'nms':>5s} {'≥2框':>6s} {'均值':>5s} {'含8B所选(召回)':>14s} {'所选一致':>9s}")
    for floor in FLOORS:
        for nms_iou in NMS_IOUS:
            n2, ns, hit, agree, tot = 0, [], 0, 0, 0
            for qid in sorted(tg["control"]):
                key = f'{q[qid]["visible"]}||{tg["control"][qid].lower()}'
                rec = raw.get(key)
                if rec is None:
                    continue
                tab = table(rec, floor, nms_iou)
                if len(tab) >= 2:
                    n2 += 1
                ns.append(len(tab))
                tot += 1
                b8 = base[qid]["bbox"]
                if any(F.iou(b, b8) >= F.IOU_KEEP for b in tab):
                    hit += 1
                new = pick(q[qid]["query"], tab)
                if new is not None and F.iou(new, b8) >= F.IOU_KEEP:
                    agree += 1
            if not tot:
                continue
            print(f"{floor:6.2f} {nms_iou:5.2f} {n2:6d} {statistics.mean(ns):5.1f} "
                  f"{hit}/{tot} = {hit/tot:5.1%}   {agree/tot:5.1%}")
    print("\n判读：召回高 = GD 表能找到 8B 找到的那个实例；'所选一致'高 = 两个枚举器"
          "\n      指向同一实例。两者都只是下限检验——目标组（模型数不清处）不可由此外推。")


def target(raw, tg, q, base, floor, nms_iou) -> None:
    print(f"\n===== 2) 目标组改动面（GD 表 floor={floor} nms={nms_iou}）=====")
    st = Counter()
    chg = []
    for qid in sorted(tg["target"]):
        key = f'{q[qid]["visible"]}||{tg["target"][qid].lower()}'
        rec = raw.get(key)
        if rec is None:
            st["未采到"] += 1
            continue
        tab = table(rec, floor, nms_iou)
        if len(tab) < 2:
            st[f"仍 <2 框"] += 1
            continue
        new = pick(q[qid]["query"], tab)
        if new is None:
            st["前提不满足"] += 1
            continue
        if F.iou(new, base[qid]["bbox"]) >= F.IOU_KEEP:
            st["与现答案一致"] += 1
            continue
        st["★会改变"] += 1
        chg.append((qid, tab, new))
    for k, v in st.most_common():
        print(f"  {k:14s} {v}")
    n = len(chg)
    print(f"  → 会改 {n} 条 | Δ 上限 {n/5690:.4f} | ×21~30% 期望 ≈ {n*0.21/5690:.4f}~{n*0.30/5690:.4f}")
    return chg


def render(chg, q, base, n_show) -> None:
    from PIL import Image, ImageDraw
    VIZ.mkdir(exist_ok=True)
    shown = 0
    for qid, tab, new in chg:
        if shown >= n_show:
            break
        try:
            im = Image.open(SEMI / q[qid]["visible"]).convert("RGB")
        except Exception:
            continue
        im = im.resize((im.width * 2, im.height * 2))
        d = ImageDraw.Draw(im)
        W, H = im.size
        for i, b in enumerate(tab, 1):
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H],
                        outline=(0, 120, 255), width=2)
            d.text((b[0] * W + 3, b[1] * H + 3), str(i), fill=(0, 120, 255))
        d.rectangle([new[0] * W, new[1] * H, new[2] * W, new[3] * H],
                    outline=(0, 220, 0), width=6)
        o = base[qid]["bbox"]
        d.rectangle([o[0] * W, o[1] * H, o[2] * W, o[3] * H],
                    outline=(255, 0, 200), width=3)
        im.save(VIZ / f"{qid}.png")
        print(f"  {qid}  GD表 {len(tab)} 框  k={F.ordinal_index(q[qid]['query'])}  "
              f"{q[qid]['query'][:52]}")
        shown += 1
    print(f"已存 {shown} 张 → {VIZ}（蓝=GD候选带序号 绿粗=所选第k框 品红=现答案）")


def build(chg, base, stem) -> None:
    import hashlib
    import subprocess
    import zipfile
    out = {k: dict(v) for k, v in base.items()}
    for qid, _t, new in chg:
        out[qid]["bbox"] = new
    (SP.out_path(stem, ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    p = SP.out_path(stem, ".zip")
    payload = json.dumps(out, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo(f"{stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, payload)
    print(f"\n已写 {p.name}  改动 {len(chg)} 条  SHA256 "
          f"{hashlib.sha256(p.read_bytes()).hexdigest().upper()[:16]}…")
    r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"), str(p)],
                       capture_output=True, text=True)
    print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                              if ln.strip().startswith(("[", "✓", "✗"))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="result_semi_enum_trunc",
                    help="基线提交件（默认已交 0.7598 的截断层产物）")
    ap.add_argument("--floor", type=float, default=0.20)
    ap.add_argument("--nms", type=float, default=0.3)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--zip", action="store_true")
    ap.add_argument("--stem", default="result_semi_gd_enum")
    a = ap.parse_args()
    raw, tg, q, cores = load()
    base = json.loads(SP.find(a.base).read_text(encoding="utf-8"))
    print(f"GD 原始 {len(raw)} 对 | 目标 {len(tg['target'])} | 控制 {len(tg['control'])}"
          f" | 基线 {a.base}")
    if a.sweep:
        sweep(raw, tg, q, cores, base)
    chg = []
    if a.target or a.render or a.zip:
        chg = target(raw, tg, q, base, a.floor, a.nms)
    if a.render:
        render(chg, q, base, a.render)
    if a.zip:
        build(chg, base, a.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
