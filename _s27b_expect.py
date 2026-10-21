#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""估计 annotate 版（10-05 复跑）相对 ref 版（10-04 基准，对方称 0.8092）的预期得分。

思路：两版同码同配置，仅解码随机性差异 ⇒ 651 条差异位上两版"谁对"对称，
期望净翻转 = 0，得分期望 ≈ 基准分；方差只来自差异位。
按 IoU 分层收窄方差：
  · IoU≥0.5 的差异位：两版同对/同错 ⇒ 不翻转（零方差贡献）
  · IoU<0.5 的差异位：可能翻转，逐位 ±1 计方差
"""
from __future__ import annotations

import json
import math
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
PKG = R / "复赛提交材料文件夹" / "参赛团队编号-团队名称-复赛-队长姓名" \
    / "参赛团队编号-代码与数据"
ANNO = PKG / "data_outputs" / "predictions_27b_annotate.json"
REF = PKG / "code27b" / "predictions_27b_ref.json"
FB = [0.49, 0.49, 0.51, 0.51]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if (ar + br - inter) > 0 else 0.0


def main() -> int:
    a = json.loads(ANNO.read_text("utf-8"))
    r = json.loads(REF.read_text("utf-8"))
    diff = [k for k in a if a[k] != r[k]]
    safe = [k for k in diff if _iou(a[k], r[k]) >= 0.5]
    risk = [k for k in diff if _iou(a[k], r[k]) < 0.5]
    fb_a = sum(1 for k in a if a[k] == FB)
    fb_r = sum(1 for k in r if r[k] == FB)
    print(f"总 {len(a)}｜差异 {len(diff)}｜同侧(IoU≥0.5) {len(safe)}"
          f"｜翻转风险面(IoU<0.5) {len(risk)}")
    print(f"回退框：ref {fb_r} ｜ annotate {fb_a}")
    base = 0.8092
    n = len(a)
    # 情形1（保守上界）：风险位逐位独立对称 ⇒ sd = sqrt(2*q*m) 量级
    m = len(risk)
    sd_entries = math.sqrt(m)          # 每风险位最多贡献 ±1 的一半
    print(f"\n期望得分 ≈ {base:.4f}（基准分；差异位对称无偏）")
    print(f"  1σ ≈ ±{sd_entries/n:.4f}（风险面 {m} 条，独立对称极限）")
    print(f"  95% 区间 ≈ [{base - 1.96*sd_entries/n:.4f}, "
          f"{base + 1.96*sd_entries/n:.4f}]")
    lo = (base * n - m) / n
    hi = (base * n + m) / n
    print(f"  绝对边界（风险位全反）：[{lo:.4f}, {hi:.4f}]")
    # 回退框差（几乎必错）的直接扣减
    if fb_a != fb_r:
        print(f"  ★回退框差 {fb_a - fb_r} 条（几乎必错）⇒ 期望再 "
              f"{'-' if fb_a > fb_r else '+'}{abs(fb_a-fb_r)/n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
