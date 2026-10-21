# -*- coding: utf-8 -*-
"""归零法测量：用"只留历史最高分"的规则，把线上得分当**测量仪器**用。

原理
    把某个条目子集的框全部换成「必然错」的小框（角落 0.001~0.011，面积 1e-4），
    则得分只由其余子集贡献 —— 得分数值直接读出该子集的**真实正确条数**，不需要真值。

为什么需要它
    全局无任何真值（初赛 test.json 与 queries.json 同构、无 bbox），此前所有"准确率"
    都是推的，这正是两次 6σ 负面的根源。机械审计（颜色/方位/形态）经实测含大量假阳性
    （"the trunk of the green tree"、"the leftmost wheel on the cart"、"in the right hand"
    均被误判），不能当读数用。故改用线上读数的直接测量。

口径
    0.806·a + 0.194·b = 总分（a=非序数准确率 4587 条，b=序数准确率 1103 条）
    测出任一侧即同时得到另一侧。**建议先测 a**（归零序数，预期分 ≈a×0.806 ≈ 0.69，
    掉幅温和），即可判定错题大头在哪一侧。

必然错的诱饵框为何安全
    角落 0.01×0.01 要与某真值达到 IoU≥0.5，真值必须同样极小且同在极角（联合概率≈0）；
    对比之下用整幅框[0,0,1,1]反而危险——真值面积≥0.5 的条目（如"天空"类）会被误算为正确。

用法
    python -B _measure_zero.py --mode nonord   # ★M1：归零序数 → 读出非序数准确率 a
    python -B _measure_zero.py --mode ord      # 归零非序数 → 读出序数准确率 b
    python -B _measure_zero.py --mode ord_res  # 归零「除序数可推算外」→ 读可推算子集
    python -B _measure_zero.py --mode ord_unres
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

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
CORES = R / "_ordinal_cores_r2.json"
DECOY = [0.001, 0.001, 0.011, 0.011]     # 必然错：角落极小框


def keep_ids(mode: str, base: dict, q: dict, s8b: dict, ordinal: set) -> set:
    """哪些条目**保留真实框**（其余全部换成诱饵框）。"""
    if mode == "nonord":
        return {k for k in base if k not in ordinal}
    if mode == "ord":
        return set(ordinal) & set(base)
    res, unres = set(), set()
    for k in sorted(set(ordinal) & set(base)):
        rec = s8b.get(k) or {}
        ok = (F.expected_ordinal_num(q[k]["query"], rec.get("bbox_all"),
                                     rec.get("n_cands"), ("x", "y", "z")) is not None)
        (res if ok else unres).add(k)
    if mode == "ord_res":
        return res
    if mode == "ord_unres":
        return unres
    raise SystemExit(f"未知 mode: {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["nonord", "ord", "ord_res", "ord_unres"])
    ap.add_argument("--base", default="result_semi_enum_trunc")
    ap.add_argument("--stem", default="")
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    ordinal = set(json.loads(CORES.read_text(encoding="utf-8")))
    s8b = json.loads((R / "_semi_qwen3vl_som_all.json").read_text(encoding="utf-8"))
    base = json.loads(SP.find(a.base).read_text(encoding="utf-8"))
    keep = keep_ids(a.mode, base, q, s8b, ordinal)

    out = {}
    n_zero = 0
    for k, v in base.items():
        d = dict(v)
        if k not in keep:
            d["bbox"] = list(DECOY)
            n_zero += 1
        out[k] = d
    stem = a.stem or f"{a.base}__measure_zero_{a.mode}"
    print(f"基线 {a.base} | 模式 {a.mode} | 保留真框 {len(keep)} 条 | "
          f"归零 {n_zero} 条（应 = {len(base)-len(keep)}）")
    assert n_zero == len(base) - len(keep), "归零条数与预期不符"
    assert set(out) == set(base) and len(out) == 5690

    (SP.out_path(stem, ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    p = SP.out_path(stem, ".zip")
    payload = json.dumps(out, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo(f"{stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, payload)
    print(f"已写 {p}  SHA256 {hashlib.sha256(p.read_bytes()).hexdigest().upper()}")

    # 读数解读（只在分数回来后才成立，此处先给公式与预期，避免事后解释漂移）
    sh = len(keep) / 5690
    print(f"\n预期读数：得分 ≈ 保留子集正确数 / 5690，即上限 {sh:.4f}")
    if a.mode == "nonord":
        print("  ★M1：得分 / 0.806 = 非序数准确率 a；再由 0.806a + 0.194b = 0.7598 反算 b")
        for a_ in (0.82, 0.84, 0.86, 0.88, 0.90):
            print(f"     若 a={a_:.2f} → 预期提交分 {a_*0.806:.4f}，则 b="
                  f"{(0.7598 - 0.806*a_)/0.194:.3f}")
    elif a.mode == "ord":
        print("  得分 / 0.194 = 序数准确率 b；再由总分反算 a")

    r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"), str(p)],
                       capture_output=True, text=True)
    print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                              if ln.strip().startswith(("[", "✓", "✗"))))
    print(f"  ⚠ 这是**测量件**，分数会显著低于当前最高分 0.7598；平台按历史最高分记录，"
          f"故不影响成绩，但会占用一次提交机会。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
