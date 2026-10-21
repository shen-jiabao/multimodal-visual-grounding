# -*- coding: utf-8 -*-
"""H-impl（方案 A）实施：只替换「与当前件 IoU<0.5」的轴类条目。

细则全部取自 `_prereg_H_impl.md` §1（**实施前写定**，本脚本不得改动其语义）：
    S1 作用域   = _prereg_H §2 的 144 条（154 经 R1–R5 后）
    S2 枚举成功 = n_cands ≥ 2
    S3 非平局   = 主口径 x1/y1、ε=0.02 下唯一
    S4 ★过滤    = 与当前件（0.7786）IoU < 0.5 才替换
    S5 替换值   = 主口径几何最左/最右候选
    S6 ★面积守卫 = 候选面积 > 0.35 或 > 1.25×当前面积 ⇒ 保留当前框（沿用项目既有守卫）
    S7 其余一切逐位不动

用法：python -B _h_impl.py [--zip]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _sub_path as SP  # noqa: E402

CUR = R / "submission_zips" / "semifinal" / "0.7786" / "result_semi_n32bf16_nonord.json"
RES = R / "_axis154_enum2.json"
STEM = "result_semi_h_axis19"
N_ORD, N_NONORD = 689, 3741
EPS, MAX_AREA, GROW = 0.02, 0.35, 1.25


def iou(a, b) -> float:
    if not a or not b or len(a) != 4 or len(b) != 4:
        return 0.0
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def area(b) -> float:
    return (b[2]-b[0]) * (b[3]-b[1])


def key(word: str, b: list) -> float:
    return b[0] if word in ("leftmost", "rightmost") else b[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    d = json.loads(RES.read_text(encoding="utf-8"))
    info, out = d["info"], d["out"]
    cur = json.loads(CUR.read_text(encoding="utf-8"))

    # S1~S3
    scope = [k for k, v in info.items() if v["kind"] == "OK"]
    enum_ok = [k for k in scope if out.get(k, {}).get("n_cands", 0) >= 2]
    uniq = []
    for k in enum_ok:
        cands = out[k].get("bbox_all") or []
        w = info[k]["word"]
        if not cands:
            continue
        vals = [key(w, b) for b in cands]
        best = min(vals) if w in ("leftmost", "topmost") else max(vals)
        if sum(1 for v in vals if abs(v - best) < EPS) == 1:
            uniq.append(k)

    def pick_of(k):
        cands = out[k]["bbox_all"]
        w = info[k]["word"]
        vals = [key(w, b) for b in cands]
        best = min(vals) if w in ("leftmost", "topmost") else max(vals)
        return cands[vals.index(best)]

    # S4 + S6
    chg, guarded, listed = [], [], []
    for k in uniq:
        cb = cur[k]["bbox"]
        pk = pick_of(k)
        if iou(pk, cb) >= 0.5:            # S4：不是"换了实例"⇒ 不动
            continue
        if area(pk) > MAX_AREA or area(pk) > GROW * area(cb):   # S6
            guarded.append((k, round(area(pk), 3), round(area(cb), 3)))
            continue
        chg.append(k)
        listed.append((k, info[k]["word"], round(iou(pk, cb), 3),
                       round(area(pk), 4), round(area(cb), 4)))

    print(f"作用域 S1 {len(scope)}｜枚举成功 S2 {len(enum_ok)}｜非平局 S3 {len(uniq)}")
    print(f"★S4 换实例（IoU<0.5） {len(chg)} 条｜S6 面积守卫另拦 {len(guarded)} 条")
    for k, w, io, ap_, ac in listed:
        print(f"    {k}  {w:9s} IoU={io:.3f}  面积 {ac} → {ap_}")
    if guarded:
        print("  面积守卫拦下：")
        for k, ap_, ac in guarded:
            print(f"    {k}  候选面积 {ap_} > min(0.35, 1.25×{ac})")

    # 实施
    final = {k: dict(v) for k, v in cur.items()}
    for k in chg:
        final[k]["bbox"] = [round(v, 6) for v in pick_of(k)]

    # 自检
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    moved_ord = [k for k in cores if final[k] != cur[k]]
    moved_other = [k for k in final if k not in cores and k not in set(chg)
                   and final[k] != cur[k]]
    assert not moved_ord, f"序数层被动 {len(moved_ord)}"
    assert not moved_other, f"作用域外非序数被动 {len(moved_other)}"
    assert len(final) == 5690 and set(final) == set(cur)
    leg = [k for k, v in final.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)}"
    print(f"\n自检：序数 {len(cores)} 逐位不动 ✓｜作用域外非序数逐位不动 ✓"
          f"｜条数 {len(final)} ✓｜bbox 合法 ✓")

    print(f"\n★ 读数口径（_prereg_H_impl §3）：")
    print(f"    得分 × 5690 − {N_ORD} = 非序数桶正确数　对 {N_NONORD}（0.7786 件）")
    print(f"    Δ = (得分 − 0.7786) × 5690")
    print(f"    上限算式（非预期）：(4430 + {len(chg)}) / 5690 ≈ "
          f"{(4430 + len(chg))/5690:.4f}")
    print(f"    ★事前预期：**不给正数**——净成功率不可观测，可能为负")

    if a.zip:
        jp = SP.out_path(STEM, ".json")
        zp = SP.out_path(STEM, ".zip")
        if jp.parent == SP.SUB:
            jp = SP.SUB / "未提交" / jp.name
            jp.parent.mkdir(exist_ok=True)
        if zp.parent == SP.SUB:
            zp = SP.SUB / "未提交" / zp.name
        jp.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
        payload = json.dumps(final, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo(f"{STEM}.json", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)
        print(f"\n已写 {jp}\n     {zp}")
        print(f"     SHA256 {hashlib.sha256(zp.read_bytes()).hexdigest().upper()}")
        r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"), str(zp)],
                           capture_output=True, text=True)
        print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                                  if ln.strip().startswith(("[", "✓", "✗"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())