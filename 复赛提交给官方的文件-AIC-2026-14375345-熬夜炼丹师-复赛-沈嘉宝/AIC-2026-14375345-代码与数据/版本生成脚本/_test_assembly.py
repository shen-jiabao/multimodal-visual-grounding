# -*- coding: utf-8 -*-
"""装配器夹具审计：构造八组受控夹具真跑一遍，逐桶核对行为与计数。

为什么要真跑而不是读代码
    装配器是出件关键路径；跑完的 4587 条若因装配错误白费，代价极大。本测试把
    "无框保留基线""序数层不动""分类互斥"等承诺变成**可判定的断言**（含桶计数）。

八组夹具
    G1 非序数·合法且远离基线   → 必须改动（桶 6）
    G2 非序数·bbox=None       → **保留基线**（桶 1）；本题核心承诺
    G3 非序数·与基线相同（小框）→ 不动（桶 5）
    G4 非序数·逆序非法         → 保留基线（桶 2）
    G5 序数·给任意框           → **保留基线**；隔离变量的关键
    G6 其余非序数（不在产物）  → 保留基线（桶 1）
    G7 非序数·合法但面积>0.35  → 保留基线（桶 3）★新增：防"大目标被误拒"看不见
    G8 非序数·基线与 32B 同为大框 → 保留基线（桶 3）★验证大框不会被误判为"一致"

用法：python -B _test_assembly.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))
import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
STEM = "ZZTEST_assembly_fixture"

q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
base = json.loads(SP.find("result_semi_enum_trunc").read_text(encoding="utf-8"))
nonord = [k for k in base if k not in cores]
ordi = [k for k in base if k in cores]
small = [k for k in nonord if F.area(base[k]["bbox"]) <= F.MAX_AREA]
big = [k for k in nonord if F.area(base[k]["bbox"]) > F.MAX_AREA]
print(f"非序数 {len(nonord)}（基线小框 {len(small)} / 基线大框 {len(big)}）| 序数 {len(ordi)}")


def far(box, dx=0.25):
    """与 box 不重叠（IoU<0.5）的合法小框。"""
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 0.05), max(y2 - y1, 0.05)
    nx1 = x1 + dx if x1 + dx <= 0.9 else 0.01
    ny1 = y1 + dx if y1 + dx <= 0.9 else 0.01
    return [round(nx1, 6), round(ny1, 6),
            round(min(nx1 + w, 0.999), 6), round(min(ny1 + h, 0.999), 6)]


G1 = small[0:20]
G2 = small[20:30]
G3 = small[30:40]
G4 = small[40:50]
G7 = small[50:60]
G8 = big[0:5]
G5 = ordi[0:10]
_groups = [set(G1), set(G2), set(G3), set(G4), set(G7), set(G8)]
assert len(set().union(*_groups)) == sum(len(g) for g in _groups), "夹具组重叠"

BIGBOX = [0.10, 0.10, 0.90, 0.75]        # 面积 0.52 > 0.35，但 ordered 合法
direct = {}
for k in G1:
    direct[k] = {"bbox": far(base[k]["bbox"]), "n": 1, "raw": "fixture"}
for k in G2:
    direct[k] = {"bbox": None, "n": 0, "raw": "[]"}
for k in G3:
    direct[k] = {"bbox": list(base[k]["bbox"]), "n": 1, "raw": "fixture"}
for k in G4:
    direct[k] = {"bbox": [0.8, 0.8, 0.2, 0.2], "n": 1, "raw": "fixture"}
for k in G7:
    direct[k] = {"bbox": list(BIGBOX), "n": 1, "raw": "fixture"}
for k in G8:
    direct[k] = {"bbox": list(base[k]["bbox"]), "n": 1, "raw": "fixture"}
for k in G5:
    direct[k] = {"bbox": [0.5, 0.5, 0.9, 0.9], "n": 1, "raw": "fixture"}

fx = R / "_fixture_direct.json"
fx.write_text(json.dumps(direct, ensure_ascii=False), encoding="utf-8")
n_missing = len(nonord) - len(G1) - len(G2) - len(G3) - len(G4) - len(G7) - len(G8)
print(f"夹具：G1={len(G1)} G2={len(G2)} G3={len(G3)} G4={len(G4)} G7={len(G7)} "
      f"G8={len(G8)} G5(序数)={len(G5)} | 未在产物中的非序数 {n_missing}")

r = subprocess.run([sys.executable, "-B", str(R / "_assembly_32b.py"),
                    "--direct", str(fx), "--stem", STEM, "--zip"],
                   capture_output=True, text=True, cwd=str(R))
print("\n----- 装配器输出 -----")
print(r.stdout)
if r.returncode != 0:
    print(r.stderr[-2000:])
    sys.exit("★ 装配器非零退出")

out = json.loads(SP.find(STEM).read_text(encoding="utf-8"))
zp = SP.find(STEM, ".zip")

fails = []


def chk(cond, msg):
    print(("  ✓ " if cond else "  ✗ ") + msg)
    if not cond:
        fails.append(msg)


print("----- 逐桶计数断言 -----")
cnt = {}
for line in r.stdout.splitlines():
    m = re.match(r"^\s+(\d)\s+(.+?)\s+(\d+)\s*$", line)
    if m:
        cnt[int(m.group(1))] = (m.group(2), int(m.group(3)))
exp = {1: len(G2) + n_missing, 2: len(G4), 3: len(G7) + len(G8),
       5: len(G3), 6: len(G1)}
print(f"  实际桶: { {k: v[1] for k, v in sorted(cnt.items())} }")
print(f"  期望桶: {exp}")
for b in (1, 2, 3, 5, 6):
    got = cnt.get(b, ("?", 0))[1]
    chk(got == exp[b], f"桶 {b}（{cnt.get(b, ('?',))[0]}）计数 {got} == 期望 {exp[b]}")
chk(sum(v[1] for v in cnt.values()) == len(nonord),
    f"各桶合计 == 非序数 {len(nonord)}")

print("----- 行为断言 -----")
chk(len(out) == 5690, f"条数 5690（实际 {len(out)}）")
chk(set(out) == set(q), "键集与 queries 一致")
chk(all(F.iou(out[k]["bbox"], direct[k]["bbox"]) > 0.999 for k in G1),
    f"G1 写出 32B 的框（{len(G1)} 条）")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G2),
    f"★G2 无框条目保留基线（{len(G2)}/{len(G2)}）")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G3), "G3 一致者未动")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G4), "G4 逆序非法保留基线")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G7),
    f"★G7 面积>0.35 保留基线（{len(G7)}/{len(G7)}）")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G8),
    f"★G8 基线与32B同为大框时保留基线（{len(G8)}/{len(G8)}）")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G5), "★G5 序数层不动")
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in ordi),
    f"★全部 {len(ordi)} 条序数与基线逐位相同")
G6 = [k for k in nonord if k not in direct]
chk(all(out[k]["bbox"] == base[k]["bbox"] for k in G6),
    f"G6 未在产物中的非序数保留基线（{len(G6)} 条）")
bad = [k for k in q if any(out[k].get(f) != q[k][f]
                          for f in ("visible", "infrared", "depth", "query"))]
chk(not bad, f"四字段原样回显（异常 {len(bad)}）")
leg = [k for k, v in out.items()
       if not (len(v["bbox"]) == 4 and 0 <= v["bbox"][0] < v["bbox"][2] <= 1
               and 0 <= v["bbox"][1] < v["bbox"][3] <= 1)]
chk(not leg, f"bbox 全合法（非法 {len(leg)}）")
with zipfile.ZipFile(zp) as zf:
    inner = json.loads(zf.read(zf.namelist()[0]).decode("utf-8"))
chk(inner == out, "zip 内 JSON 与磁盘 JSON 语义相同")
chk(zp.parent.name == "未提交", f"新产物落 未提交/（实际 {zp.parent.name}）")

print("\n===== 审计结论 =====")
print("全部通过 ✓ 装配器行为与逐桶计数均与声称一致" if not fails
      else f"★ {len(fails)} 项未通过")
for f in fails:
    print("  - " + f)
for p in (fx, SP.find(STEM), zp):
    p.unlink(missing_ok=True)
print("测试产物已清理")
sys.exit(1 if fails else 0)