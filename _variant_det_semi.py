# -*- coding: utf-8 -*-
"""从已提交的 v1 派生「序数层文本推算」变体（离线，零 GPU）。

三层，逐层叠加（每层只动上层未动的条目，故可线性相加）：
    det      一律用文本推算的候选（candidates[k-1]）替换 SoM 读出的编号
    _ss      在 det 之上，对「候选未按 x 排序」的条目先自排序再推算
    _qfb     在 det[_ss] 之上，对仍判 ORD_UNDET（四前提至少缺一）的条目
             改用 Qwen **直接框**——即 `_fuse_qwen.decide` 的 ORD_QFB 那一支

判据**全部复用** `_fuse_qwen`（`det_ordinal` / `expected_ordinal_num` /
`decide` / `ordered` / `norm` / `iou` / `valid_cand`），本脚本不另写任何判据。

用法：
    python _variant_det_semi.py --selfsort --qfb
    python _variant_det_semi.py --probe          # 只看 ORD_UNDET 的前提分解
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))
import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
SUB = R / "submission_zips" / "semifinal"

# 仅用于 `undet_probe` 的分类打印（"②方向不明"里有多少是垂直/深度轴），
# **不参与任何决策**：`_fuse_qwen.direction` 只认水平轴，此处不得越权替它判断。
VERT_PAT = re.compile(
    r"(top\s*to\s*bottom|bottom\s*to\s*top|from\s+the\s+top|from\s+the\s+bottom|"
    r"uppermost|topmost|lowermost|bottommost)", re.I)
DEPTH_PAT = re.compile(
    r"(closest|nearest|farthest|furthest|frontmost|backmost|"
    r"from\s+the\s+front|from\s+the\s+back)", re.I)


def load() -> tuple[dict, set, dict, dict, dict]:
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(encoding="utf-8")))
    d8b = json.loads((R / "_semi_qwen3vl_all.json").read_text(encoding="utf-8"))
    s8b = json.loads((R / "_semi_qwen3vl_som_all.json").read_text(encoding="utf-8"))
    v1 = json.loads(SP.find("result_final_semi").read_text(encoding="utf-8"))
    return q, cores, d8b, s8b, v1


def build(which: str, base: dict | None = None,
          selfsort: bool = False,
          qfb: bool = False,
          axes: tuple = ("x",)) -> tuple[dict, Counter, list]:
    """返回 (提交字典, 规则计数, 变更列表)。

    v1 的框结构为 {visible, infrared, depth, query, bbox}，此处**只改 bbox**，
    其余字段一律原样保留（回显字段不得改动）。

    selfsort=True 时，对「候选未按 x 排序」的条目先自行按 x 中心升序排列再交给
    `det_ordinal`。这不是放宽判据：`det` 用文本推算，不依赖模型的编号，只需要
    一份正确的左到右候选表；而 `cands_left_to_right` 本是为「编号→位置」映射把
    关。只修正输入，判据仍全部来自 `_fuse_qwen`。

    qfb=True 时，对仍判 ORD_UNDET 的条目改用 Qwen 直接框。实现上把 `qs` 传
    None 再交给 `F.decide`：这会命中 `is_ordinal` 的**回退分支**（ORD_QFB），
    而不进 SOM 分支——文本推算既已判不可靠，模型自报编号就不该再作证据，
    否则会把已被 LA 覆盖的框退回 SoM，属未预期的改动。判据本身不重写。

    axes=("x","y") 时启用垂直序数轴（`_fuse_qwen.ordinal_axis`，默认只开 x）。
    由于 x 优先于 y，已由 x 轴解出的条目在此轴下给出同一结果（IoU=1，不计改动），
    故**本层的改动集就是垂直轴增量**，与前几层不相交。
    """
    q, cores, d8b, s8b, v1 = load()
    if base is None:
        base = v1
    out = {k: dict(v) for k, v in base.items()}
    stats: Counter = Counter()
    changes: list = []

    for qid in sorted(cores):
        rec = s8b.get(qid) or {}
        old = base[qid]["bbox"]
        src = rec
        if selfsort:
            ca = rec.get("bbox_all") or []
            if ca and not F.cands_left_to_right(ca):
                src = dict(rec)
                src["bbox_all"] = sorted(ca, key=lambda b: (b[0] + b[2]) / 2)
        new, rule = F.det_ordinal(which, src, q[qid]["query"], axes)
        if new is None and qfb and rule == "ORD_UNDET":
            qd = (d8b.get(qid) or {}).get("bbox")
            new, rule = F.decide(F.DET_NONORD.get(which, which), old, qd, None,
                                 True, query=q[qid]["query"],
                                 n_cands=rec.get("n_cands"), qfb=True)
        stats[rule] += 1
        if new is None:
            continue                       # 不改动：回落基线
        if F.iou(new, old) >= F.IOU_KEEP:
            stats["与现答案一致(不改动)"] += 1
            continue
        out[qid] = dict(base[qid])
        out[qid]["bbox"] = [round(float(x), 6) for x in new]
        changes.append((qid, rule, old, out[qid]["bbox"]))
    return out, stats, changes


def undet_probe(which: str, selfsort: bool = False,
                axes: tuple = ("x",)) -> Counter:
    """诊断：把 ORD_UNDET 按「哪个前提先失败」分解，供判读下一步杠杆。

    只用于判读，**不参与任何决策**。selfsort / axes 与 `build` 同义（诊断必须与
    将要构建的层级看同一批条目，否则数字对不上）。分解顺序与
    `expected_ordinal_num` 内的判定顺序逐条对齐，并就地断言三件事，防止诊断与
    实现漂移：
      1. 每条都能归类（无"未归类"）
      2. 归类的条目确实被判为 None
      3. 已启用 y 轴时，仍落在"轴不可判"的条目**不得含垂直轴词**（否则说明
         垂直轴漏接）
    """
    q, cores, _d8b, s8b, _v1 = load()
    st: Counter = Counter()
    for qid in sorted(cores):
        rec = s8b.get(qid) or {}
        query = q[qid]["query"]
        src = rec
        if selfsort:
            ca = rec.get("bbox_all") or []
            if ca and not F.cands_left_to_right(ca):
                src = dict(rec)
                src["bbox_all"] = sorted(ca, key=lambda b: (b[0] + b[2]) / 2)
        _new, rule = F.det_ordinal(which, src, query, axes)
        if rule != "ORD_UNDET":
            continue
        st["合计"] += 1
        cands = src.get("bbox_all") or []
        n = len(cands)
        k = F.ordinal_index(query)
        ax, _ori = F.ordinal_axis(query, axes)
        if not query or not cands:
            reason = "①候选表空"
        elif src.get("n_cands") is not None and src["n_cands"] != n:
            reason = "①记录与候选表条目数不符"
        elif not ax:
            reason = "②轴不可判"
            # 仅诊断：轴不可判的里面还有多少只是缺"深度轴"
            if "y" in axes:
                if VERT_PAT.search(query):
                    st["②轴不可判-含垂直词(BUG)" ] += 1
                elif DEPTH_PAT.search(query):
                    st["②轴不可判-深度轴"] += 1
            elif VERT_PAT.search(query):
                st["②轴不可判-垂直轴(未启用)"] += 1
        elif ax == "x" and not F.cands_left_to_right(cands):
            reason = ("①候选不足2" if n < F.CAND_MIN_N else "①候选未按x排序")
        elif k is None:
            reason = "③序号不可抽取"
        elif k > n:
            reason = f"③k({k})>候选数({n})"
        elif F.ordinal_is_reference(query):
            reason = "④序数为参照物"
        else:
            reason = "??未归类"
        assert reason != "??未归类", f"{qid} 分解失败：{query}"
        assert F.expected_ordinal_num(query, cands, src.get("n_cands"), axes) is None, \
            f"{qid} 分解与 expected_ordinal_num 漂移：{reason}"
        st[reason] += 1
    return st


def package(data: dict, stem: str) -> Path:
    """打成提交 zip：内部 JSON 名 == zip 基名，固定时间戳保证容器 SHA 稳定。"""
    out = SP.out_path(stem, ".zip")
    payload = json.dumps(data, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo(f"{stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, payload)
    (SP.out_path(stem, ".json")).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {out.name}  {out.stat().st_size} B  "
          f"SHA256 {hashlib.sha256(out.read_bytes()).hexdigest().upper()[:16]}…")
    return out


def validate(path: Path) -> int:
    """独立校验（调用既有校验器，不另写）。"""
    import subprocess
    r = subprocess.run([sys.executable, "-B", str(R / "_validate_semi.py"),
                        str(path)], capture_output=True, text=True)
    print("  " + "\n  ".join(
        ln for ln in r.stdout.splitlines() if ln.strip().startswith(("[", "✓", "✗"))))
    return r.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="det", choices=["det", "det_rb"])
    ap.add_argument("--selfsort", action="store_true",
                    help="在 det 之上追加「候选自排序」扩展（改动集不相交，可叠加）")
    ap.add_argument("--yaxis", action="store_true",
                    help="再追加垂直序数轴 axes=('x','y')（改动集与前几层不相交）")
    ap.add_argument("--qfb", action="store_true",
                    help="再追加「ORD_UNDET 改用 Qwen 直接框」（改动集与前几层不相交）")
    ap.add_argument("--probe", action="store_true",
                    help="只打印 ORD_UNDET 的前提分解，不生成任何文件")
    ap.add_argument("--no-zip", action="store_true")
    a = ap.parse_args()

    # 层序固定：det → ss → y → qfb。**每层必须携带前面所有层的开关**——`build`
    # 每次都从零复核全部条目，只传本层增量会让后面的层退回旧配置（曾因此在层 4
    # 丢掉 axes，把层 3 已按垂直轴定好的框又用 direct 覆盖掉）。配置累积后，前序
    # 层动过的条目在后续层给出同一结果（IoU=1，不计改动），故改动集仍互不相交。
    flags: list[dict] = []
    if a.selfsort:
        flags.append(dict(tag="ss", selfsort=True))
    if a.yaxis:
        flags.append(dict(tag="y", axes=("x", "y")))
    if a.qfb:
        flags.append(dict(tag="qfb", qfb=True))
    plan: list[dict] = [dict(tag="")]
    acc: dict = {}
    for f in flags:
        acc = {**acc, **{k: v for k, v in f.items() if k != "tag"}}
        plan.append(dict(tag=f["tag"], **acc))

    stem0 = f"result_semi_ord_{a.which}"
    labels = [stem0]
    for step in plan[1:]:
        labels.append(f"{labels[-1]}_{step['tag']}")

    cur = json.loads(SP.find("result_final_semi").read_text(encoding="utf-8"))

    if a.probe:
        last = {k: v for k, v in plan[-1].items() if k != "tag"}
        print(f"===== ORD_UNDET 前提分解（{a.which}，已应用 {labels[-1]}，"
              f"axes={last.get('axes', ('x',))}）=====")
        for i, step in enumerate(plan[:-1]):
            cur, _s, _c = build(a.which, base=cur,
                                **{k: v for k, v in step.items() if k != "tag"})
        st = undet_probe(a.which, selfsort=last.get("selfsort", False),
                         axes=last.get("axes", ("x",)))
        for name, n in st.most_common():
            print(f"  {name:26s} {n}")
        return 0

    for i, step in enumerate(plan):
        kw = {k: v for k, v in step.items() if k != "tag"}
        data, stats, chg = build(a.which, base=cur, **kw)
        print(f"===== 层 {i+1}/{len(plan)}: {labels[i]}"
              f"（基线 = {labels[i-1] if i else 'result_final_semi'}，配置 {kw}）=====")
        print(f"  规则计数: {dict(stats)}")
        print(f"  本层改动 {len(chg)} 条")
        for qid, rule, old, new in chg[:5]:
            print(f"    {qid}  {rule}  旧{[round(x, 4) for x in old]} → 新{[round(x, 4) for x in new]}")
        cur = data
        if not a.no_zip:
            print(f"  校验退出码 {validate(package(cur, labels[i]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
