# -*- coding: utf-8 -*-
"""序数层装配（冻结稿 `_frozen_criteria.md` 的 SoM 层）。

基线：**当前最高分件** `result_semi_n32_ordbase`（线上 0.7742）
作用域：冻结稿 §2 定的 **407 条**（不可推算；口径在 8B 表上定死，见
        `_ordinal_cores_unres.json`）
闸门：冻结稿 §4 三条——**全过才替换，缺一保留原框**
  1. 候选表 ≥ 2 项
  2. 方向与 k 可抽取，且 k ≤ 候选数
  3. 选中框合法（有序 + 面积 ∈ [MIN_AREA, MAX_AREA]）

★隔离变量：**非序数 4587 条与作用域外的序数一律逐位不动**。故本件得分可直接读出
  序数层的净翻转：`得分 × 5690 − 4405 = Δ序数`（4405 = 非序数桶 3730 + 序数 675）。

事前预期（写在此处，避免事后解释漂移）
  改动面 66 条；先验用**实测**净成功率 **17.6%**（来自 32B 非序数件 82/467，
  见 `_frozen_criteria__跑后实录.md` §6.1）⇒ 预期 **+12 条 → ≈ 0.7763**
  这是一个**上界性质的估计**：66 条 × 17.6%，样本小（σ≈3），须以线上读数为准。

用法
    python -B _assembly_ord_som.py [--zip]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
BASE = "result_semi_n32_ordbase"        # ★基线 = 当前最高分件（0.7742）
SCOPE = R / "_ordinal_cores_unres.json"  # 冻结稿 §2 的 407 条
SOM = R / "_qwen32b_som_scoped.json"     # 32B SoM（407 作用域）
AXES = ("x", "y")
N_NONORD_OK = 3730                       # 实测：非序数桶正确数
N_ORD_OK = 675                           # 实测：现有序数层正确数


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="result_semi_n32_ordsom")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    scope = json.loads(SCOPE.read_text(encoding="utf-8"))
    som = json.loads(SOM.read_text(encoding="utf-8"))
    base = json.loads(SP.find(BASE).read_text(encoding="utf-8"))
    print(f"基线 {BASE}（{len(base)} 条，线上 0.7742）")
    print(f"作用域 {len(scope)} 条（不可推算）| 32B SoM 覆盖 "
          f"{sum(1 for k in scope if k in som)}")

    out = {k: dict(v) for k, v in base.items()}
    st = Counter()
    changes = []
    for k in sorted(scope):
        rec = som.get(k) or {}
        nb = rec.get("bbox")
        n = rec.get("n_cands") or 0
        ax, _o = F.ordinal_axis(q[k]["query"], AXES)
        kk = F.ordinal_index(q[k]["query"])
        # 三条闸门（冻结稿 §4）——缺一保留原框
        if n < 2:
            st["1 候选表<2 → 保留原框"] += 1
            continue
        if not (bool(ax) and kk is not None and kk <= n):
            st["2 方向/k 不可抽 → 保留原框"] += 1
            continue
        if not (nb and F.valid_cand(nb)):
            st["3 选中框非法 → 保留原框"] += 1
            continue
        nb = F.norm(nb)
        if F.iou(nb, base[k]["bbox"]) >= F.IOU_KEEP:
            st["4 与基线一致(IoU≥0.5) → 不动"] += 1
            continue
        st["5 ★改动"] += 1
        changes.append((k, base[k]["bbox"], nb, F.iou(nb, base[k]["bbox"])))
        out[k]["bbox"] = nb

    for k, v in sorted(st.items()):
        print(f"  {k:28s} {v}")
    tot = sum(st.values())
    assert tot == len(scope), f"分类总和 {tot} ≠ 作用域 {len(scope)}（分类不互斥）"
    print(f"  合计 {tot} = 作用域 {len(scope)} ✓")

    # ---- 硬自检 ----
    assert len(out) == 5690, len(out)
    assert set(out) == set(q), "键集与 queries 不一致"
    nonord = [k for k in base if k not in cores]
    bad_n = [k for k in nonord if out[k] != base[k]]
    assert not bad_n, f"非序数层被动过 {len(bad_n)} 条（必须逐位不动）"
    ord_same = [k for k in base if k in cores and out[k] != base[k]]
    assert len(ord_same) == len(changes), \
        f"序数层改动 {len(ord_same)} ≠ 预期 {len(changes)}"
    bad = [k for k in q if any(out[k].get(f) != q[k][f]
                               for f in ("visible", "infrared", "depth", "query"))]
    assert not bad, f"字段回显异常 {len(bad)} 条"
    leg = [k for k, v in out.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)} 条"
    print(f"\n自检：非序数 4587 条逐位不动 ✓｜序数层改动 {len(ord_same)} 条 = 桶 5 ✓")
    print("自检：条数/键集/字段回显/bbox 合法性 —— 全部通过")

    n = len(changes)
    print(f"\n★ 改动 {n} 条 × 实测先验 17.6% ≈ +{n * 0.176:.0f} 条 → "
          f"≈ {0.7742 + n * 0.176 / 5690:.4f}（事前预期，非读数）")
    print(f"  读数口径（★据实）：得分×5690 − {N_NONORD_OK + N_ORD_OK} = Δ序数翻转"
          f"（{N_NONORD_OK}=实测非序数桶，{N_ORD_OK}=实测序数层）")
    for k, o, nb, io in changes[:6]:
        print(f"    {k} IoU(与基线)={io:.3f}  {q[k]['query'][:50]}")

    if a.zip:
        jp, zp = SP.out_path(a.stem, ".json"), SP.out_path(a.stem, ".zip")
        if jp.parent == SP.SUB:
            jp = SP.SUB / "未提交" / jp.name
            jp.parent.mkdir(exist_ok=True)
        if zp.parent == SP.SUB:
            zp = SP.SUB / "未提交" / zp.name
        jp.write_text(json.dumps(out, ensure_ascii=False, indent=2),
                      encoding="utf-8")
        payload = json.dumps(out, ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            zi = zipfile.ZipInfo(f"{a.stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, payload)
        print(f"\n已写 {jp}")
        print(f"     {zp}\n     SHA256 "
              f"{hashlib.sha256(zp.read_bytes()).hexdigest().upper()}")
        r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"),
                            str(zp)], capture_output=True, text=True)
        print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                                  if ln.strip().startswith(("[", "✓", "✗"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())