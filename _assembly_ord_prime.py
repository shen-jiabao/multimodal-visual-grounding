# -*- coding: utf-8 -*-
"""序数层装配（预登记 B′）：`32B 直接框` 替换**同机制**基线。

判据与预期**全部写定在** `_prereg_B_prime.md`，本脚本只做实现，不含任何新判据。

作用范围（可检验条件）
    序数 1103 条中，`IoU(基线, 8B直接框) ≥ 0.999` 者 ⇒ 343 条。
    这是「可迁移的不是规则文本，而是**被替换那一层的性质**」这条教训的直接实现。

闸门（全过才替换，缺一保留基线）
    1. 32B 有框  2. `valid_cand`（有序 + 面积 ∈ [MIN_AREA, MAX_AREA]）  3. 与基线 IoU < 0.5

事前预期（见预登记 §4，写定后不得改）
    改动面 87 → × 实测先验 17.6% ≈ **+15 条 → ≈0.7769**（基线 0.7742）
    ⚠ 本层只承诺"机制匹配"，不承诺量级（17.6% 测自较易的非序数层，本作用域是最难子集）。

用法
    python -B _assembly_ord_prime.py [--zip]
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
BASE = "result_semi_n32_ordbase"          # 缺省输出基线（B′：线上 0.7742）
D8B = R / "_semi_qwen3vl_all.json"        # 8B 直接框（判定"基线来源"的参照）
D32 = R / "_qwen32b_direct_ord.json"      # 32B 直接框（含 1103 序数）
SAME = 0.999                              # 基线 = 8B 直接框 的判定阈值（预登记 §1）
PRIOR = 0.176                             # 实测边际成功率（同机制）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="result_semi_n32_orddirect")
    ap.add_argument("--base", default=BASE, help="输出基线件（缺省=B′ 的 0.7742 件）")
    ap.add_argument("--ref", default=None, help="作用域参照件（缺省同 --base）")
    ap.add_argument("--lo", type=float, default=SAME, help="作用域下界（含）")
    ap.add_argument("--hi", type=float, default=1.0000001, help="作用域上界（不含）")
    ap.add_argument("--score", type=float, default=0.7742, help="基线线上分（仅用于打印）")
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    ref = a.ref or a.base

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    base = json.loads(SP.find(a.base).read_text(encoding="utf-8"))
    prev = json.loads(SP.find(ref).read_text(encoding="utf-8"))
    d8b = json.loads(D8B.read_text(encoding="utf-8"))
    d32 = json.loads(D32.read_text(encoding="utf-8"))
    print(f"输出基线 {a.base}（{len(base)} 条，线上 {a.score}）| 作用域参照 {ref}")
    print(f"作用域判据：{a.lo} ≤ IoU(参照框, 8B直接框) < {a.hi}")

    out = {k: dict(v) for k, v in base.items()}
    st = Counter()
    changes = []
    scope = []
    for k in sorted(cores):
        b8 = (d8b.get(k) or {}).get("bbox")
        if not b8:
            continue
        io8 = F.iou(prev[k]["bbox"], F.norm(b8))
        if not (a.lo <= io8 < a.hi):
            continue                       # 基线不是 8B 直接框 ⇒ 非同机制，按预登记排除
        scope.append(k)
        nb = (d32.get(k) or {}).get("bbox")
        if not nb:
            st["1 32B 无框 → 保留基线"] += 1
            continue
        nb = F.norm(nb)
        if not F.valid_cand(nb):
            st["2 框非法 → 保留基线"] += 1
            continue
        if F.iou(nb, base[k]["bbox"]) >= F.IOU_KEEP:
            st["3 与基线一致(IoU≥0.5) → 不动"] += 1
            continue
        st["4 ★改动"] += 1
        changes.append((k, base[k]["bbox"], nb, F.iou(nb, base[k]["bbox"])))
        out[k]["bbox"] = nb

    print(f"作用范围: {len(scope)} 条")
    for k, v in sorted(st.items()):
        print(f"  {k:28s} {v}")
    tot = sum(st.values())
    assert tot == len(scope), f"分类总和 {tot} ≠ 作用范围 {len(scope)}（分类不互斥）"
    print(f"  合计 {tot} = 作用范围 {len(scope)} ✓")

    # ---- 硬自检（与预登记承诺逐条对应）----
    assert len(out) == 5690, len(out)
    assert set(out) == set(q), "键集与 queries 不一致"
    nonord = [k for k in base if k not in cores]
    bad_n = [k for k in nonord if out[k] != base[k]]
    assert not bad_n, f"非序数层被动过 {len(bad_n)} 条（必须逐位不动）"
    ord_chg = [k for k in base if k in cores and out[k] != base[k]]
    assert len(ord_chg) == len(changes), \
        f"序数层改动 {len(ord_chg)} ≠ 预期 {len(changes)}"
    outside = [k for k in ord_chg if k not in set(scope)]
    assert not outside, f"改动落在作用范围之外 {len(outside)} 条（越界）"
    bad = [k for k in q if any(out[k].get(f) != q[k][f]
                               for f in ("visible", "infrared", "depth", "query"))]
    assert not bad, f"字段回显异常 {len(bad)} 条"
    leg = [k for k, v in out.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)} 条"
    print(f"\n自检：非序数 4587 条逐位不动 ✓｜序数改动 {len(ord_chg)} 条 = 桶 4 ✓"
          f"｜全部落在作用范围内 ✓")
    print("自检：条数/键集/字段回显/bbox 合法性 —— 全部通过")

    n = len(changes)
    bc = round(a.score * 5690)         # 基线件的总正确条目数（由线上分反算，勿硬编码）
    print(f"\n★ 改动 {n} 条 × 实测先验 {PRIOR:.1%} ≈ +{n * PRIOR:.0f} 条 → "
          f"≈ {a.score + n * PRIOR / 5690:.4f}（预登记的事前预期，非读数）")
    print(f"  读数口径：得分×5690 − {bc} = Δ（{bc} 由基线线上分 {a.score} 反算）")
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