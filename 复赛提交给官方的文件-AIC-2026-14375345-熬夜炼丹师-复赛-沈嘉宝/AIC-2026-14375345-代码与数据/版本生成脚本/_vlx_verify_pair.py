# -*- coding: utf-8 -*-
"""验证"路线一件"的单变量隔离：两件必须**只差 646 层**，其余逐位相同。

为什么必须先验这个
    两次得分之差 ΔA − ΔB 只有在"两件除 646 层外完全一致"时才等于 646 层的净翻转。
    构造保证（装配器自检）不等于已核验——本项目有明文规矩：
    **"装配脚本审计需构造受控夹具"、"硬断言不能替代已核验"**。
    故对两个**成品件**做逐位比对。

用法
    python -B _vlx_verify_pair.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import _sub_path as SP  # noqa: E402

SUP = R / "_vlx_sup_split.json"


def main() -> int:
    base = json.loads(SP.find("result_semi_n32_orddirect").read_text(
        encoding="utf-8"))
    sup = set(json.loads(SUP.read_text(encoding="utf-8")))
    nonord = [k for k in base if k not in set(json.loads(
        (R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))]
    rest = [k for k in nonord if k not in sup]
    print(f"基线 {len(base)} 条｜非序数 {len(nonord)}｜"
          f"最高级层 {len(sup)}｜其余 {len(rest)}")

    def load(stem):
        return json.loads(SP.find(stem).read_text(encoding="utf-8"))

    a = load("result_semi_vlx_nonord")      # A：两层都改
    b = load("result_semi_vlx_rest3941")    # B：只改 3941

    ok = True

    # ① 646 层：B 必须与**基线**逐位相同（这几条在本次提交里没被改）
    diff_sup_b = [k for k in sup if b[k] != base[k]]
    print(f"\n① 646 层 {len(sup)} 条：B 与基线逐位相同 → "
          f"{'✓ 一致' if not diff_sup_b else f'✗ 有 {len(diff_sup_b)} 条不同'}")
    ok &= not diff_sup_b

    # ② 646 层：A 与 B 应当不同（A 改了这层）
    diff_sup_ab = [k for k in sup if a[k] != b[k]]
    print(f"② 646 层：A 与 B 不同 → {len(diff_sup_ab)} 条"
          f"（即该层真正被改动的条数，装配器报 139）")
    ok &= len(diff_sup_ab) == 139

    # ③ 3941 层：A 与 B 必须逐位相同（单变量隔离的核心）
    diff_rest = [k for k in rest if a[k] != b[k]]
    print(f"③ 3941 层 {len(rest)} 条：A 与 B 逐位相同 → "
          f"{'✓ 一致' if not diff_rest else f'✗ 有 {len(diff_rest)} 条不同'}")
    ok &= not diff_rest

    # ④ 序数层：两件都应与基线逐位相同
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(
        encoding="utf-8")))
    diff_ord = [k for k in cores if a[k] != base[k] or b[k] != base[k]]
    print(f"④ 序数层 {len(cores)} 条：两件均与基线逐位相同 → "
          f"{'✓ 一致' if not diff_ord else f'✗ 有 {len(diff_ord)} 条不同'}")
    ok &= not diff_ord

    print(f"\n隔离性结论：{'✓ 通过——两次得分之差可归因于 646 层' if ok else '✗ 未通过'}")
    print("\n读数分解（B 的得分记为 S_B）：")
    print("    Δ_3941 = round(S_B × 5690) − 689 − 3730")
    print("    Δ_646  = −156 − Δ_3941")
    print("  —— 例：S_B = 0.7773 ⇒ Δ_3941 = +4，Δ_646 = −160")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
