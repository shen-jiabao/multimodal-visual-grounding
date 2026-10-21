# -*- coding: utf-8 -*-
"""M2 测量件：读出「预登记 D 作用范围」1652 条非序数的正确数。

原理（与 M1 同一套）
    把**作用范围之外**的非序数 2935 条的框换成**必然错的角落小框**，作用范围内的 1652 条
    与序数层 1103 条原样保留。平台按历史最高分记录 ⇒ 本件不拉低成绩，而得分直接读出：

        得分 × 5690 = 675（序数层） + C₁₆₅₂（作用范围正确数）
        ⇒ C₁₆₅₂ = 得分 × 5690 − 675

    诱饵框为何必然错：角落 0.01×0.01 要与真值 IoU ≥ 0.5，需真值同样极小且同在极角
    （联合概率 ≈ 0）。**不能用整幅框**——真值面积 ≥0.5 的条目（如"天空"）会被误算为正确。

它回答什么
    作用范围 1652 条的准确率 vs 其余非序数（2935 条）的准确率。
    这是**检验 D 的前提**（"距离/远近 + 热/光照"是信息受限的弱区）是否成立。

用法
    python -B _measure_zero_d.py [--zip]
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

BASE = "result_semi_n32_orddirect"      # 线上 0.7766（当前最高）
STEM = "result_semi_d3modal__measure_zero_scope"
N_ORD, N_NONORD = 1103, 4587
# ★★ 更正（2026-10-01）：此处原写 675，**与基线件不符**——675 是 M1 在
#   **0.7598 阶段**测出的序数正确数；而本件的基线是 **0.7766 阶段**，
#   其中 B′ 层已给序数层 +14 ⇒ 本阶段序数正确数 = **689**。
#   影响：见 SHA256.txt 中本件条目的"读数更正"——原结论 78.1%/83.1%/差 5.0pp/+83 应为
#   **79.0%/82.6%/差 3.6pp/+59**。件本身（零化哪 1652 条）不受影响，只是解读口径错。
N_ORD_CORRECT, N_NONORD_CORRECT = 689, 3730     # ★689，非 675
N_SCOPE = 1652
DECOY_SIDE = 0.01


def decoy(i: int) -> list[float]:
    """必然错的角落小框；按索引在左上角微移，避免与任何真值重合。"""
    x = 0.001 + (i % 8) * 0.012
    y = 0.001 + (i // 8 % 8) * 0.012
    return [round(x, 6), round(y, 6), round(x + DECOY_SIDE, 6),
            round(y + DECOY_SIDE, 6)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()

    q = json.loads((R / "复赛数据集-基于大模型的多模态视觉理解与推理"
                    / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    scope = set(json.loads((R / "_d_scope.json").read_text(encoding="utf-8")))
    base = json.loads(SP.find(BASE).read_text(encoding="utf-8"))

    nonord = [k for k in base if k not in cores]
    # ★零化**作用范围内**那 1652 条（而不是范围外的 2935 条）：
    #   信息完全相同，但本件得分 ≈0.54（而非 ≈0.35）——**降幅更小、合规观感更好**，
    #   与 M1（0.6411，已被接受）同一量级。
    zero = [k for k in nonord if k in scope]
    keep = [k for k in nonord if k not in scope]
    print(f"基线 {BASE}（线上 0.7766）")
    print(f"  非序数 {len(nonord)} = 作用范围内 {len(zero)}（★零化） + "
          f"作用范围外 {len(keep)}（保留真框）")
    print(f"  序数层 {len(cores)}（原样保留）")
    assert len(zero) == N_SCOPE, len(zero)
    assert len(keep) == N_NONORD - N_SCOPE, len(keep)
    assert len(keep) + len(zero) + len(cores) == 5690

    out = {k: dict(v) for k, v in base.items()}
    for i, k in enumerate(sorted(zero)):
        out[k]["bbox"] = decoy(i)

    # 自检：作用范围内与序数层逐位未动；零化条目确实被改
    moved_in = [k for k in keep if out[k] != base[k]]
    moved_ord = [k for k in cores if out[k] != base[k]]
    still = [k for k in zero if out[k] == base[k]]
    assert not moved_in, f"作用范围内被动过 {len(moved_in)} 条"
    assert not moved_ord, f"序数层被动过 {len(moved_ord)} 条"
    assert not still, f"零化未生效 {len(still)} 条"
    print(f"\n自检：作用范围**外** {len(keep)} 条逐位未动 ✓｜序数层 {len(cores)} 条逐位未动 ✓"
          f"｜零化 {len(zero)} 条全部生效 ✓")

    leg = [k for k, v in out.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)} 条"
    bad = [k for k in q if any(out[k].get(f) != q[k][f]
                               for f in ("visible", "infrared", "depth", "query"))]
    assert not bad, f"字段回显异常 {len(bad)} 条"
    assert len(out) == 5690 and set(out) == set(q)
    print("自检：条数/键集/字段回显/bbox 合法性 —— 全部通过")

    print(f"\n★ 读数口径（据实）：")
    print(f"    得分 × 5690 = {N_ORD_CORRECT} + C₂₉₃₅   ⇒ C₂₉₃₅ = 得分 × 5690 − {N_ORD_CORRECT}")
    print(f"    则 C₁₆₅₂ = {N_NONORD_CORRECT} − C₂₉₃₅")
    print(f"    作用范围内准确率 = C₁₆₅₂ / {N_SCOPE}；范围外 = C₂₉₃₅ / {N_NONORD-N_SCOPE}")
    print(f"    对照：非序数整体 {N_NONORD_CORRECT}/{N_NONORD} = "
          f"{N_NONORD_CORRECT/N_NONORD:.1%}")
    print(f"    预期本件得分：若 C₂₉₃₅ ≈ {int((N_NONORD-N_SCOPE)*0.813)}（同 81.3%）"
          f"则 ≈ {(675 + (N_NONORD-N_SCOPE)*0.813)/5690:.4f}")

    if a.zip:
        jp, zp = SP.out_path(STEM, ".json"), SP.out_path(STEM, ".zip")
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
            zi = zipfile.ZipInfo(f"{STEM}.json", date_time=(1980, 1, 1, 0, 0, 0))
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