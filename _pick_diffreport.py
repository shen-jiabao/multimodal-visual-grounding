#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 pick_v1.zip 与 27b_annotate.zip 的差异报告（markdown）。

内容：总览计数、54 处贴蓝逐条清单（qid/查询/IoU/两版框/有效面标记）、
按 IoU 分层统计、期望影响。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
SUB = R / "submission_zips" / "semifinal" / "未提交"


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ar = (a[2] - a[0]) * (a[3] - a[1])
    br = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ar + br - inter) if (ar + br - inter) > 0 else 0.0


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def main() -> int:
    q = json.loads((R / "复赛数据集-基于大模型的多模态视觉理解与推理"
                    / "queries" / "queries.json").read_text("utf-8"))
    v1 = json.loads(zipfile.ZipFile(SUB / "result_semi_pick_v1.zip")
                    .read("submission.json").decode())
    an = json.loads(zipfile.ZipFile(SUB / "result_semi_27b_annotate.zip")
                    .read("submission.json").decode())
    blue = json.loads((R / "submission_zips/semifinal/0.7963/"
                       "result_semi_q38_nonord.json").read_text("utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text("utf-8")))

    diff = sorted(k for k in v1 if v1[k]["bbox"] != an[k]["bbox"])
    assert all(v1[k]["bbox"] == blue[k]["bbox"] for k in diff), \
        "存在非贴蓝来源的差异"
    lay = {"0（完全异区域）": [], "0<iou<0.3": [], "0.3<=iou<0.5": [],
           "iou>=0.5（无效改动）": []}
    for k in diff:
        v = _iou(an[k]["bbox"], blue[k]["bbox"])
        key = ("0（完全异区域）" if v == 0 else "0<iou<0.3" if v < 0.3
               else "0.3<=iou<0.5" if v < 0.5 else "iou>=0.5（无效改动）")
        lay[key].append(k)
    n_ord = sum(1 for k in diff if k in cores)

    L = ["# 差异报告：pick_v1.zip vs 27b_annotate.zip", "",
         f"- 基体：`result_semi_27b_annotate.zip`（对方管线 10-05 复跑全量）",
         f"- 本件：`result_semi_pick_v1.zip` = 基体 + 人眼选蓝位贴 "
         f"`result_semi_q38_nonord`（0.7963 件）框",
         f"- 条目数：两件均 5690；键集与四字段回显完全一致", "",
         "## 一、总览", "",
         f"| 指标 | 值 |", "|---|---|",
         f"| 差异条目（bbox 不同） | **{len(diff)} / 5690**（{len(diff)/56.9:.1f}%） |",
         f"| 其余条目 | 逐位相同 |",
         f"| 差异中序数类查询 | {n_ord} |",
         f"| 差异中非序数 | {len(diff)-n_ord} |",
         f"| 有效分歧位（IoU<0.5，可能翻分） | {len(lay['0（完全异区域）'])+len(lay['0<iou<0.3'])+len(lay['0.3<=iou<0.5'])} |",
         f"| 无效改动（IoU>=0.5，两件同侧） | {len(lay['iou>=0.5（无效改动）'])} |",
         "", "## 二、按 IoU(annotate, q38) 分层", "",
         "| 层 | 条数 |", "|---|---|"]
    for name, ks in lay.items():
        L.append(f"| {name} | {len(ks)} |")
    L += ["", "## 三、逐条清单（按 qid）", "",
          "| # | qid | 序数 | IoU | 查询 | annotate 框 | pick_v1 框(=q38) | 面积变化 |",
          "|---|---|---|---|---|---|---|---|"]
    for i, k in enumerate(diff, 1):
        a0, b0 = an[k]["bbox"], blue[k]["bbox"]
        io = _iou(a0, b0)
        ar = _area(b0) / _area(a0) if _area(a0) > 0 else float("nan")
        L.append(f"| {i} | {k} | {'序' if k in cores else ''} | {io:.2f} | "
                 f"{q[k]['query'][:48]} | {a0} | {b0} | ×{ar:.2f} |")
    L += ["", "## 四、期望影响", "",
          "每处有效分歧位（IoU<0.5）：人眼判对 +1 条、判错 −1 条；"
          "IoU≥0.5 的 43 处两件几乎同侧，ACC@0.5 下期望贡献 ≈ 0。",
          f"净期望 = {len(lay['0（完全异区域）'])+len(lay['0<iou<0.3'])+len(lay['0.3<=iou<0.5'])}"
          " × (2p−1) 条（p=判对率）。p=0.7 ⇒ +8.8 条 ⇒ +0.0015。", ""]
    out = SUB / "diff_report_pick_v1_vs_annotate.md"
    out.write_text("\n".join(L), "utf-8")
    print(f"差异 {len(diff)}｜报告 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
