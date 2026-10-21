# -*- coding: utf-8 -*-
"""消融候选审计：提交前的差异矩阵 + 功效判据（见 `消融实验复盘.md` M1/M2）。

用途
- 算任意两版之间的【总差异 / 实质差异 / 微差】，并给出每个版本的最近邻
- 用 `0.335 * sqrt(D)` 判定一次提交是否**能产生知识**（D < 36 → 不产生）
- 只在**能干净分离自变量**时才计算效率（实质或微差其一为 0），
  混合对比一律拒绝给出效率 —— 这是复盘 §3.3 的纪律，写进代码而非留在文档

用法：
    python _audit_ablation.py                      # 全矩阵 + 聚类 + 最近邻
    python _audit_ablation.py --policy det_full_qfb_nosom   # 审计单个候选

依赖：仅标准库 + `_fuse_qwen`（复用 iou / norm / IOU_KEEP，避免判据漂移）
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _fuse_qwen as F  # noqa: E402

R = Path(r"d:\Desktop\2026AiC")
BASE = "基线"
# 已实测的线上得分（新增提交后需补录，否则不计算效率）
SCORES = {
    "基线": 0.6695, "safe": 0.7075, "shrink": 0.7082, "qwen": 0.7590,
    "full": 0.7606, "det_rb": 0.7613, "det": 0.7692, "det_full": 0.7709,
    "det_full_qfb": 0.7795, "det_full_qfb_wide": 0.7798,
    "det_full_qfb_nosom": 0.7827,
}
POLICIES = ("qwen", "safe", "shrink", "full", "det", "det_rb", "det_full",
            "det_full_qfb", "det_full_qfb_wide", "det_full_qfb_nosom",
            "det_full_qfb_nosom_qonly")
NEAR = 200        # 近邻阈值（实质差异条数），复盘 §2.2
MIN_D = 36        # 功效判据临界值：D < 36 → 不产生知识（复盘 §4.1）
# 差异有两层，杠杆相差约 100 倍（复盘 §3），故功效必须**两层分别贡献**。
# 只算实质差异会漏判「纯微差」对比 —— 例：qwen↔full 实质差 0、微差 5486，
# 实测 +15 条（2.8σ）确为可测；旧判据会误报"无差异"。以下常量即为此修正。
EFF_SUB, EFF_MIC = 0.30, 0.0028      # 各自的经验净成功率
FLIP_SUB, FLIP_MIC = 0.80, 0.0056    # 各自的 (P增益 + P损失)
MIC_MIN = 2857                       # 纯微差可测所需最少条数（由上式解出）
ENUM_MAX = 20                        # 差异 ≤ 该值 → 「枚举型实验」，见 verdict()


def power(sub: int, mic: int) -> tuple[float, float]:
    """返回 (可区分性 σ 倍数, 预期效应**幅度**)。

    只给幅度，**不给方向**：方向取决于"新候选是往哪边改"，本工具不推断。
    例：微差层已测的 0.28% 增益方向是【保留基线】（`qwen`→`full`），故与之
    反向的候选预期为负 —— 若工具默认给正号，就会得出相反结论。
    """
    exp = EFF_SUB * sub + EFF_MIC * mic
    var = FLIP_SUB * sub + FLIP_MIC * mic
    return (exp / math.sqrt(var) if var > 0 else 0.0), exp


def direction_hint(sub: int, mic: int) -> str:
    """给出方向推断提示（仅作方向提示）。"""
    if sub == 0 and mic > 0:
        return ("纯微差：已测增益方向为【保留基线】；若本候选是【放弃基线】"
                "，则该效应预期为负")
    if sub > 0 and mic == 0:
        return "纯实质差异：方向须引用同类型已测结论（实测区间 +10.8%~+39.6%）"
    return "混合差异：方向须引用同类型已测结论，本工具不推断"


def verdict(sub: int, mic: int) -> str:
    """按复盘 §4.3/§4.4 给出结论档。

    两类实验不可混谈：
    - **统计型**（差异大、条目同质）：关注"这类改动整体是否有效" → 需 σ ≥ 2
    - **枚举型**（差异极少、条目可逐条定位）：关注"这几条的具体翻转" → 任意规模
      都有确定信息。线上得分是**确定性**的，σ 描述的是"哪些条目翻转"的未知，
      不是测量噪声；故 D 条改动的结果直接给出净值 = 赢数 − 输数
    """
    total = sub + mic
    if total == 0:
        return "无任何差异"
    if total <= ENUM_MAX:
        return (f"枚举型实验（差异 {total} 条 ≤ {ENUM_MAX}，可逐条定位）："
                "结果给出净值整数，可直接判读，不受 σ<2 限制")
    p, _ = power(sub, mic)
    if p < 2:
        return "★ 不产生知识（不消耗额度）"
    if sub >= NEAR:
        return "强实验"
    if sub < MIN_D:
        return f"可测（靠微差：微差需 ≥{MIC_MIN} 条）→ 须提交前写定判读档位"
    return "可测（须提交前写定判读档位）"


def load(name: str) -> dict | None:
    """读入某策略产物；`基线` 读第二代锁定版。文件缺失返回 None。"""
    if name == BASE:
        p = R / "result_locate_merge_som.json"
    else:
        p = R / f"result_qwen_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def diff(a: dict, b: dict) -> tuple[int, int]:
    """返回 (总差异, 实质差异)。实质 = 数值不同且 IoU<IOU_KEEP。"""
    tot = sub = 0
    for q in a:
        if F.norm(a[q]["bbox"]) == F.norm(b[q]["bbox"]):
            continue
        tot += 1
        if F.iou(a[q]["bbox"], b[q]["bbox"]) < F.IOU_KEEP:
            sub += 1
    return tot, sub


def bbox_ok(b) -> bool:
    """bbox 是否有序且落在 [0,1] 内（x1<x2, y1<y2）。"""
    x1, y1, x2, y2 = b
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def verify(policy: str) -> int:
    """提交前硬约束校验（清单阶段 B1）。返回 0 全过，1 有未通过项。"""
    import hashlib
    import zipfile

    pj = R / f"result_qwen_{policy}.json"
    pz = R / f"result_qwen_{policy}.zip"
    base = load(BASE)
    if not pj.exists():
        print(f"  ✗ 缺产物 {pj.name}")
        return 1
    data = json.loads(pj.read_text(encoding="utf-8"))
    test = json.loads(F.TEST_JSON.read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

    bad_field = sum(1 for v in data.values() if set(v) != F.FIELDS)
    bad_bbox = sum(1 for v in data.values() if not bbox_ok(v["bbox"]))
    n_err = sum(1 for v in data.values() if v.get("error"))
    frozen = sum(1 for q in F.FROZEN
                 if F.norm(data[q]["bbox"]) != F.norm(base[q]["bbox"]))
    inner = None
    if pz.exists():
        zf = zipfile.ZipFile(pz)
        inner = hashlib.sha256(zf.read(zf.namelist()[0])).hexdigest()
    # hexdigest() 返回小写，sha() 返回大写，须统一后再比较
    in_ok = bool(inner) and inner.lower() == sha(pj).lower()

    rows = [
        ("条数 == 9555", len(data) == 9555, str(len(data))),
        ("键集 == test.json", set(data) == set(test), str(len(set(data)))),
        ("字段集 == FIELDS", bad_field == 0, f"异常 {bad_field}"),
        ("bbox 有序且 ∈[0,1]", bad_bbox == 0, f"异常 {bad_bbox}"),
        ("无 error 条目", n_err == 0, f"error {n_err}"),
        ("冻结 6 条零改动", frozen == 0, f"被改 {frozen}"),
        ("zip 内 JSON == 磁盘", in_ok,
         "一致" if in_ok else ("无 zip" if not inner else "不一致")),
    ]
    print(f"提交前硬约束校验：{policy}")
    ok = True
    for name, passed, detail in rows:
        ok &= bool(passed)
        print(f"  {'✓' if passed else '✗'} {name:22s} {detail}")
    print(f"\n  JSON SHA256 {sha(pj)}")
    if pz.exists():
        print(f"  ZIP  SHA256 {sha(pz)}")
    print(f"\n  {'全部通过 → 可进入阶段 B4' if ok else '存在未通过项 → 不得提交'}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", help="只审计该候选（对其最近邻）")
    ap.add_argument("--verify", metavar="POLICY", help="提交前硬约束校验")
    args = ap.parse_args()

    if args.verify:
        sys.exit(verify(args.verify))

    names = [BASE] + list(POLICIES)
    data = {n: load(n) for n in names}
    have = [n for n in names if data[n] is not None]

    pairs = {p: diff(data[p[0]], data[p[1]])
             for p in itertools.combinations(have, 2)}

    def g(a: str, b: str) -> tuple[int, int]:
        return pairs[(a, b)] if (a, b) in pairs else pairs[(b, a)]

    if args.policy:
        tgt = args.policy
        others = [n for n in have if n != tgt]
        near = min(others, key=lambda o: sum(g(tgt, o)))
        tot, sub = g(tgt, near)
        mic = tot - sub
        p, exp = power(sub, mic)
        print(f"候选 {tgt}")
        print(f"  最近邻 {near}：总差异 {tot}  实质 {sub}  微差 {mic}")
        print(f"  可区分性 {p:.2f}σ  →  {verdict(sub, mic)}")
        print(f"  预期效应幅度 ≈ {exp:.0f} 条（方向见下，本工具不推断正负）")
        print(f"  方向提示：{direction_hint(sub, mic)}")
        best = max((n for n in have if n in SCORES), key=lambda n: SCORES[n])
        if best in others:
            bt, bs = g(tgt, best)
            bm = bt - bs
            bp, be = power(bs, bm)
            print(f"\n  vs 当前最高分 {best}（{SCORES[best]}）："
                  f"总 {bt} / 实质 {bs} / 微差 {bm}")
            print(f"     → {bp:.2f}σ，效应幅度 {be:.0f} 条；"
                  f"分数影响 = ±{be / 9555:.4f}（符号按 {direction_hint(bs, bm)} 判断）")
        print("\n  与全部已测版本（按实质差异升序）：")
        for o in sorted(others, key=lambda o: g(tgt, o)[1]):
            t2, s2 = g(tgt, o)
            print(f"    vs {o:22s} 实质 {s2:5d}  微差 {t2 - s2:5d}")
        return

    print(f"已载入 {len(have)} 个版本\n")
    print("=== 两两实质差异矩阵 ===")
    print(" " * 21 + "".join(f"{n[:9]:>10s}" for n in have))
    for a in have:
        row = f"{a[:20]:20s} "
        for b in have:
            row += f"{'—':>10s}" if a == b else f"{g(a, b)[1]:>10d}"
        print(row)

    par = {n: n for n in have}

    def find(x: str) -> str:
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in itertools.combinations(have, 2):
        if g(a, b)[1] <= NEAR:
            ra, rb = find(a), find(b)
            if ra != rb:
                par[ra] = rb
    cl = defaultdict(list)
    for n in have:
        cl[find(n)].append(n)

    print(f"\n=== 单链聚类（阈值 {NEAR} 条）===")
    for i, v in enumerate(sorted(cl.values(), key=lambda x: -len(x)), 1):
        d = max((g(a, b)[1] for a, b in itertools.combinations(v, 2)),
                default=0)
        s = ", ".join(f"{n}={SCORES[n]}" for n in v if n in SCORES)
        print(f"  簇{i}（{len(v)}，直径 {d}）：{v}\n      {s}")
    print(f"\n  {len(have)} 个版本 → {len(cl)} 个家族")

    print("\n=== 效率（只在能干净分离时给出）===")
    for a, b in itertools.combinations(have, 2):
        tot, sub = g(a, b)
        mic = tot - sub
        if a not in SCORES or b not in SCORES:
            continue
        if not (sub == 0 or mic == 0):     # 混合对比：拒绝计算（纪律）
            continue
        d = round((SCORES[b] - SCORES[a]) * 9555)
        kind = "微差" if sub == 0 else "实质"
        n = mic if sub == 0 else sub
        print(f"  {a} → {b}：纯{kind} {n} 条 → {d:+d} 条"
              f"（效率 {d / n:.2%}）")
    print("  （其余对比为实质+微差混合，无法分解 → 按纪律不计算）")


if __name__ == "__main__":
    main()
