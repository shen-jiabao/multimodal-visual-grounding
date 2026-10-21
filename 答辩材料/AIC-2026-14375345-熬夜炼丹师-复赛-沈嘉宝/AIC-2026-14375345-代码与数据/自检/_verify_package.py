# -*- coding: utf-8 -*-
"""交付包自检：一次性核验本包的全部关键主张（离线、纯 CPU、仅用标准库）。

用法
    python _verify_package.py
    （可在任意目录运行；脚本按自身位置定位包根）

检查项
    [1] 关键文件指纹：交付件 / 中间产物 / 查询索引 / 环境实录 / 技术报告 /
        判据预登记 / 远端现场证据 / 装配底座 —— 逐一比对 SHA256
    [2] 交付件硬约束：条数 5690 / 键集与 queries 一致 / 四字段原样回显 / bbox 全合法
    [3] 改动面读数：交付件相对 0.7786 基线的真分歧条数应为 519（判据 IoU < 0.5）
    [4] 结论汇总与退出码（0 = 全过；1 = 有失败）

说明：本脚本只读，不修改任何文件。
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent      # .../AIC-2026-14375345-代码与数据/自检
ROOT = HERE.parent                          # .../AIC-2026-14375345-代码与数据
PKG = ROOT.parent                           # .../AIC-2026-14375345-熬夜炼丹师-复赛-沈嘉宝

# (相对包根的路径, 期望 SHA256, 说明)
EXPECT = [
    ("AIC-2026-14375345-赛题结果文档/result_semi_q38_nonord.zip",
     "FB6C2E471A4F5E8907318FF540A5CAABAEC17A29ECAFC17EB651F512A4AE8F35",
     "★交付件（线上 0.7963）"),
    ("AIC-2026-14375345-代码与数据/_qwen38_direct.json",
     "E561606CAC8ECC7ECC4A0DEE7E7F087289365BCE95C37074577E9D1E648CCBF0",
     "交付件装配输入（Qwen3.8-27B 非序数直出）"),
    ("AIC-2026-14375345-代码与数据/_ordinal_cores_r2.json",
     "CFBA587D75AE3FB53CD64BFCC5E684EB81369C6B67AEA022F4D7B945DAEB80C2",
     "序数核心词表（作用域判据）"),
    ("AIC-2026-14375345-代码与数据/复赛数据集-基于大模型的多模态视觉理解与推理/queries/queries.json",
     "717D15A1B9C270BEFA3B5930D0B9B00EF9C567C91BD0D54819EF0E17DE5795C4",
     "查询索引（键集来源）"),
    ("AIC-2026-14375345-代码与数据/env_record.txt",
     "132B7271840BF94B8DCF020084937C9E75ED3F3C6623AE0FECF7DF6F3C9E5F22",
     "环境实录（STAGE-1/2）"),
    ("AIC-2026-14375345-技术方案.PDF",
     "500F6FDF6E24BFC2CC317865E5EFC8C5501788604577EB1FA227806BB98B658C",
     "技术方案 / 技术报告 PDF"),
    ("AIC-2026-14375345-模型文件/model_sha256.txt",
     "9DD8BF0077BD495EDB3DDCF81C944A4D48676911F569D4774B2491853420CEF8",
     "权重逐文件 SHA256 清单"),
    ("AIC-2026-14375345-代码与数据/判据预登记/_prereg_I.md",
     "28D2F255C085C327A64E51C29C32F96F10233916968A05B9885D76AF2615F099",
     "★交付件路线的判据预登记（写于推理之前）"),
    ("AIC-2026-14375345-代码与数据/判据预登记/_frozen_criteria.md",
     "D4A4BC357FAD0F1EE9AD136F756BFEB26B4CF7E949A95E11125BB8389C9E6AE1",
     "冻结稿（判据总纲）"),
    ("AIC-2026-14375345-代码与数据/远端现场证据/远端原件脚本/_qwen_full_pipeline.py",
     "6A91DC2BEB6077830143C809C9906C21DA3DD3D46E38D6BF93EEA9C781E10D5B",
     "主管线（远端原件，含 /root 硬编码路径）"),
    ("AIC-2026-14375345-代码与数据/远端现场证据/运维脚本归档/_get_serve_q38.sh",
     "409DCD334C59181EEFD163A645A0FD8639359477234C97D198314D062430AAC9",
     "起服脚本（三处同源：工作区归档 = 43504 实例 = 本包）"),
    ("AIC-2026-14375345-代码与数据/远端现场证据/_q38_gate20.json",
     "12B08F0558D74246AAD39F6E6D25CFB0A7BCB1217728043169839489BB445DA7",
     "闸门 G2 冒烟产物（20 条原始输出）"),
    ("AIC-2026-14375345-代码与数据/过程记录/改动面可视化/_sheet_changed.png",
     "C163F56D77653EEADAAC316BE731F6DB9DA3F569DB46F664202C0D01797E9863",
     "改动面拼版（20 处改动样例）"),
    ("AIC-2026-14375345-代码与数据/submission_zips/semifinal/0.7786/result_semi_n32bf16_nonord.json",
     "20B96A42F9C43BD941323476B96533D03B38FD7EDF4F5091EC9B8693B8A1443F",
     "★装配底座（0.7786 基线，供 [3] 复算 519）"),
]

ZIP_REL = EXPECT[0][0]
BASE_REL = EXPECT[-1][0]
SEMI = ROOT / "复赛数据集-基于大模型的多模态视觉理解与推理"
FIELDS = ("visible", "infrared", "depth", "query")
EXPECT_DIVERGE = 519


def sha256(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            d.update(blk)
    return d.hexdigest()


def ordered(b) -> bool:
    """bbox 合法：四元、数值、非 NaN、0~1、x1<x2 且 y1<y2。"""
    if not (isinstance(b, list) and len(b) == 4):
        return False
    if not all(isinstance(x, (int, float)) and x == x for x in b):
        return False
    if not all(0.0 <= x <= 1.0 for x in b):
        return False
    return b[0] < b[2] and b[1] < b[3]


def iou(a, c) -> float:
    ix = max(0.0, min(a[2], c[2]) - max(a[0], c[0]))
    iy = max(0.0, min(a[3], c[3]) - max(a[1], c[1]))
    inter = ix * iy
    uni = ((a[2] - a[0]) * (a[3] - a[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter)
    return inter / uni if uni > 0 else 0.0


def main() -> int:
    hard = []

    print("=" * 78)
    print("交付包自检 ——", PKG.name)
    print("=" * 78)

    print("\n[1] 关键文件指纹")
    for rel, want, note in EXPECT:
        p = PKG / rel
        if not p.exists():
            print(f"  ✗ 缺失            {rel}")
            hard.append(f"缺失 {rel}")
            continue
        got = sha256(p).upper()
        want = want.upper()
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {'一致' if ok else '不一致'}  {p.name}   [{note}]")
        if not ok:
            print(f"      期望 {want}\n      实际 {got}")
            hard.append(f"指纹不符 {rel}")

    print("\n[2] 交付件硬约束")
    zpath = PKG / ZIP_REL
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        data = json.loads(zf.read(names[0]).decode("utf-8"))
    queries = json.loads((SEMI / "queries" / "queries.json").read_text("utf-8"))

    print(f"  [2.1] 条数        {len(data)} / 应 {len(queries)}", end="  ")
    if len(data) == len(queries):
        print("✓")
    else:
        print("✗")
        hard.append("条数不符")

    print("  [2.2] 键集与 queries 一致", end="  ")
    if set(data) == set(queries):
        print("✓")
    else:
        print("✗")
        hard.append("键集不符")

    nf = necho = 0
    for k, v in data.items():
        if set(v) != set(FIELDS) | {"bbox"}:
            nf += 1
            continue
        if any(v[f] != queries[k][f] for f in FIELDS):
            necho += 1
    print(f"  [2.3] 四字段原样回显  异常 {nf} / 回显不一致 {necho}", end="  ")
    print("✓" if nf == 0 and necho == 0 else "✗")
    if nf or necho:
        hard.append("字段回显问题")

    bad = [k for k, v in data.items() if not ordered(v.get("bbox"))]
    print(f"  [2.4] bbox 全合法  非法 {len(bad)}", end="  ")
    print("✓" if not bad else "✗")
    if bad:
        hard.append("bbox 非法")

    print("\n[3] 改动面读数（相对 0.7786 基线）")
    base = json.loads((PKG / BASE_REL).read_text("utf-8"))
    n_div = sum(1 for k in data
                if iou(data[k]["bbox"], base[k]["bbox"]) < 0.5)
    ok3 = n_div == EXPECT_DIVERGE
    print(f"  真分歧条数 = {n_div}（应 {EXPECT_DIVERGE}）", end="  ")
    print("✓" if ok3 else "✗")
    if not ok3:
        hard.append(f"改动面读数 {n_div} != {EXPECT_DIVERGE}")

    print("\n" + "=" * 78)
    if hard:
        print("结论：✗ 未通过")
        for h in hard:
            print("   ✗", h)
        return 1
    print("结论：✓ 全部通过（指纹 14/14、硬约束 4/4、改动面 519）")
    print(f"交付件指纹：{EXPECT[0][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
