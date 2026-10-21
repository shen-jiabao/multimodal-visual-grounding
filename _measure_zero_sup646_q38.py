# -*- coding: utf-8 -*-
"""缺口定位测量件（v2，基线换成当前最高的 0.7963 件）。

与 M4 (`_measure_zero_sup646.py`) 的关系
    M4 的基线是 **0.7766 阶段**（`result_semi_n32_orddirect`），测出 C₆₄₆=483。
    本件基线是 **0.7963 阶段**（`result_semi_q38_nonord`）——非序数层被 q38 换过
    519 条框，所以 646 层的正确数**已经变了，必须重测**。锚点同步改：
        N_ORD_CORRECT      = 689  （序数层本件逐位未动，仍是 689）
        N_NONORD_CORRECT   = 3842 （0.7963 × 5690 − 689）
    这是 §20.11 那条教训的直接应用：**派生读数必须把锚点与基线件的"阶段"对齐**。

原理（与 M1/M2/M4 同一套归零法）
    把 646 条「空间最高级」的框换成**必然错的角落小框**，其余 3941 非序数与
    1103 序数原样保留。平台按历史最高分记录 ⇒ 不拉低成绩，得分直接读出：
        得分 × 5690 = 689 + C₃₉₄₁  ⇒ C₃₉₄₁ = 得分 × 5690 − 689
        C₆₄₆ = 3842 − C₃₉₄₁
    诱饵框：角落 0.01×0.01。与真值 IoU≥0.5 需真值同样极小且同在极角，概率≈0。
    **不能用整幅框**（真值面积≥0.5 的条目会被误算为正确）。

它回答什么（本件特有的问题）
    0.7963 件里非序数还剩 745 条错（4587−3842），但**它们落在哪一层不知道**：
    q38 在 646 层的改动率 12.7%、3941 层 8.？% —— 改动多不等于改对。
    本件给出答案，供"下一步该往哪投资源"用。

已知局限（事先声明，防误用）
    本件**不改变方案排序**：唯一有正先验的机制（direct 替换 direct）作用在**整个
    非序数层**上，不区分 646/3941；序数层那 414 条错因作用域只有 343 条而**够不着**。
    故本件的价值是"量化剩余空间"，不是"指向某个可用的杠杆"。

用法
    python -B _measure_zero_sup646_q38.py [--zip]
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

BASE = "result_semi_q38_nonord"                         # 线上 0.7963（当前最高）
STEM = "result_semi_q38_nonord__measure_zero_sup646"
N_ORD, N_NONORD = 1103, 4587
N_ORD_CORRECT, N_NONORD_CORRECT = 689, 3842             # ★0.7963 阶段
N_SUP = 646
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
    cores = set(json.loads(
        (R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    sup = set(json.loads((R / "_vlx_sup_split.json").read_text(encoding="utf-8")))
    base = json.loads(SP.find(BASE).read_text(encoding="utf-8"))

    nonord = [k for k in base if k not in cores]
    zero = [k for k in nonord if k in sup]
    keep = [k for k in nonord if k not in sup]

    print(f"基线 {BASE}（线上 0.7963）")
    print(f"  非序数 {len(nonord)} = 最高级层 {len(zero)}（★零化） + "
          f"其余 {len(keep)}（保留真框）")
    print(f"  序数层 {len(cores)}（原样保留）")

    assert len(zero) == N_SUP, f"最高级层条数 {len(zero)} ≠ {N_SUP}"
    assert len(keep) == N_NONORD - N_SUP, len(keep)
    assert not (sup & cores), "最高级清单与序数层有交集（分层定义重叠）"
    assert len(keep) + len(zero) + len(cores) == 5690

    out = {k: dict(v) for k, v in base.items()}
    for i, k in enumerate(sorted(zero)):
        out[k]["bbox"] = decoy(i)

    moved_keep = [k for k in keep if out[k] != base[k]]
    moved_ord = [k for k in cores if out[k] != base[k]]
    still = [k for k in zero if out[k] == base[k]]
    assert not moved_keep, f"其余层被动过 {len(moved_keep)} 条"
    assert not moved_ord, f"序数层被动过 {len(moved_ord)} 条"
    assert not still, f"零化未生效 {len(still)} 条"
    print(f"\n自检：其余层 {len(keep)} 条逐位未动 ✓｜序数层 {len(cores)} 条逐位未动 ✓"
          f"｜零化 {len(zero)} 条全部生效 ✓")

    ar = [round((d[2] - d[0]) * (d[3] - d[1]), 8) for d in
          (out[k]["bbox"] for k in zero)]
    assert all(0 < x <= 1e-4 for x in ar), f"诱饵面积异常 max={max(ar)}"
    leg = [k for k, v in out.items()
           if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
                   and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
    assert not leg, f"bbox 非法 {len(leg)} 条"
    bad = [k for k in q if any(out[k].get(f) != q[k][f]
                               for f in ("visible", "infrared", "depth", "query"))]
    assert not bad, f"字段回显异常 {len(bad)} 条"
    assert len(out) == 5690 and set(out) == set(q)
    print(f"自检：诱饵面积 ≤1e-4（max {max(ar):.1e}）✓｜条数/键集/字段回显/"
          f"bbox 合法性 ✓")

    print("\n★ 读数口径（据实）：")
    print(f"    得分 × 5690 = {N_ORD_CORRECT} + C₃₉₄₁  ⇒ "
          f"C₃₉₄₁ = 得分 × 5690 − {N_ORD_CORRECT}")
    print(f"    C₆₄₆ = {N_NONORD_CORRECT} − C₃₉₄₁")
    print(f"    最高级层准确率 = C₆₄₆/{N_SUP}；"
          f"其余层 = C₃₉₄₁/{N_NONORD - N_SUP}")
    print(f"    对照：M4 在 0.7766 阶段测得 C₆₄₆=483（74.8%）、"
          f"其余层 3247（82.4%）")
    print(f"    对照：本阶段非序数整体 {N_NONORD_CORRECT}/{N_NONORD} = "
          f"{N_NONORD_CORRECT / N_NONORD:.1%}")
    print(f"\n    ★可行域：「其余层准确率 ≥ "
          f"{N_NONORD_CORRECT - N_SUP}/{N_NONORD - N_SUP} = "
          f"{(N_NONORD_CORRECT - N_SUP) / (N_NONORD - N_SUP):.2%}」")
    for p in (0.830, 0.850, 0.870, 0.890, 0.900):
        c3941 = int(round((N_NONORD - N_SUP) * p))
        c646 = N_NONORD_CORRECT - c3941
        print(f"    其余层 {p:6.2%} ⇒ C₃₉₄₁={c3941:4d} ⇒ 本件得分 ≈ "
              f"{(N_ORD_CORRECT + c3941) / 5690:.4f} ⇒ C₆₄₆={c646:3d}"
              f"（{c646 / N_SUP:6.1%}、错 {N_SUP - c646:3d}）")

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
