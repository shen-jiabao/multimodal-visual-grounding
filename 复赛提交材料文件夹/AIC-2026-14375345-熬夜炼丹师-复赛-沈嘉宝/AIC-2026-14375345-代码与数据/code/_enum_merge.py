# -*- coding: utf-8 -*-
"""枚举重跑结果的本地合并、保真度校验与变体生成（判据全部复用 `_fuse_qwen`）。

远端只采集原始框（`_enum_raw.json`），一切取舍都在这里做——改策略不必再占 GPU。

四件事
1. `--regress`  **p1 回归测试**：p1 框 vs 原 `bbox_all`，逐条比对。若不一致，说明重跑
   环境有偏差，后续任何对比都不成立（本项是所有比较的前提）。
2. `--fidelity` **控制组保真度**：控制组是"原候选数已 ≥ k、文本推算已生效"的条目。
   比较「原表所选框」与「合并表所选框」，差异率过高即合并引入系统性扰动。
3. `--target`   目标集改动面：合并表相对当前最优提交会改动多少条、最大 Δ 多少。
4. `--render N` 目标样例目检图：合并候选（蓝、带序号）+ 所选第 k 框（绿粗）。

策略
    A  仅 p1（单提示基线，应与原表一致）
    B  p1+p2+p3（仅全文提示）
    C  p1+p2+p3+t1..t4（全合奏）
    D  以 p1 为基，只补 p1 完全未覆盖的框（others 中与 p1 全部框 IoU<0.5 者）——索引偏移最小

用法：
    python -B _enum_merge.py --regress --fidelity --target --strategy C
"""
from __future__ import annotations
# ===== 提交包路径适配（打包时自动注入；算法逻辑未改动）=====
import os as _os
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
ROOT = _Path(_os.environ.get("AIC_ROOT") or _HERE.parent)
ROOT_S = str(ROOT)
# ===== 注入结束 =====

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

R = Path(ROOT_S)
sys.path.insert(0, str(R))
import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
SUB = R / "submission_zips" / "semifinal"
RAW = R / "_enum_raw.json"
TARGETS = R / "_enum_targets.json"
BASE_JSON = SUB / "result_semi_ord_det_ss_y_qfb.json"
AXES = ("x", "y")
DEDUP_IOU = 0.9      # 合并去重阈值
D_P1_IOU = 0.5       # 策略 D：判断"p1 未覆盖"的阈值
E_CLUSTER_IOU = 0.3  # 策略 E：碎片聚类的 IoU 阈值（与 frag() 同口径）
TRUNC_MIN_N = 14     # 截断组：原表框数达到此值即视为可能被 512-token 上限截断
VIZ = R / "viz_enum_retry"
KS = ("p1", "p2", "p3", "t1", "t2", "t3", "t4")
BASE: dict = {}      # 当前最优提交件内容，由 load() 填充；_write_zip 用它统计改动条数


def dedup(boxes: list, thr: float = DEDUP_IOU) -> list:
    """按 IoU 贪心去重；只保留合法框。"""
    out = []
    for b in boxes:
        if not F.ordered(b):
            continue
        if any(F.iou(b, o) > thr for o in out):
            continue
        out.append(list(b))
    return out


def merge(rec: dict, strategy: str) -> list:
    """按策略合并一对 (图, core) 的原始框；返回左到右排序后的候选表。

    合并后统一按 x 中心升序排序——与 `_ss` 层同理：序号由问句文本推算，不依赖
    模型给出的编号，只需一份沿正确轴有序的候选表（**修正输入，非放宽判据**）。
    """
    if strategy == "A":
        boxes = list(rec.get("p1") or [])
    elif strategy == "B":
        boxes = [b for k in ("p1", "p2", "p3") for b in (rec.get(k) or [])]
    elif strategy == "C":
        boxes = [b for k in KS for b in (rec.get(k) or [])]
    elif strategy == "D":
        p1 = dedup(rec.get("p1") or [])
        extra = [b for k in ("p2", "p3", "t1", "t2", "t3", "t4")
                 for b in (rec.get(k) or [])]
        boxes = p1 + [b for b in extra
                      if not any(F.iou(b, o) > D_P1_IOU for o in p1)]
    elif strategy == "E":
        # 全合奏，但**按 IoU0.3 聚类后每簇只留一个代表框**——直接对治碎片化。
        # 代表框优先取 p1 的框（原管线所用提示，可信基线），否则取面积最大者。
        allb = dedup([b for k in KS for b in (rec.get(k) or [])])
        p1 = dedup(rec.get("p1") or [])
        n = len(allb)
        par = list(range(n))

        def find(i: int) -> int:
            while par[i] != i:
                par[i] = par[par[i]]
                i = par[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                if F.iou(allb[i], allb[j]) > E_CLUSTER_IOU:
                    a, b = find(i), find(j)
                    if a != b:
                        par[a] = b
        groups: dict = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(allb[i])
        boxes = []
        for members in groups.values():
            hit = next((m for m in members
                        if any(F.iou(m, o) > DEDUP_IOU for o in p1)), None)
            boxes.append(hit or max(members, key=F.area))
    else:
        raise ValueError(strategy)
    return sorted(dedup(boxes), key=lambda b: (b[0] + b[2]) / 2)


def pick(query: str, table: list):
    """在候选表上按问句推算取框；不可推算/非法时返回 None。

    与 `_fuse_qwen.det_ordinal` 同源：先 `expected_ordinal_num` 取序号，
    再在 `ordered_candidates` 的第 exp 项取值。此处 `n_cands` 传 None，因为
    候选数来自我们自己的合并，不是模型的声明值。
    """
    if not table:
        return None
    exp = F.expected_ordinal_num(query, table, None, AXES)
    if exp is None:
        return None
    seq = F.ordered_candidates(query, table, AXES)
    if seq is None or exp > len(seq):
        return None
    b = seq[exp - 1]
    return F.norm(b) if F.valid_cand(b) else None


def frag(table: list, thr: float = 0.3) -> tuple[int, int]:
    """返回 (框数, 簇数)：按 IoU>thr 并查集聚类。

    用途是识别"同一实例被切成多个框"——碎片化会让"第 k 个"错位，而这一点
    单看候选数变多是看不出来的。簇数/框数 越接近 1 越干净。
    """
    n = len(table)
    par = list(range(n))

    def find(i: int) -> int:
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if F.iou(table[i], table[j]) > thr:
                a, b = find(i), find(j)
                if a != b:
                    par[a] = b
    return n, len({find(i) for i in range(n)})


def load(base_stem: str = "result_semi_ord_det_ss_y_qfb"):
    """载入全部输入。`base_stem` 指定基线提交件——分层构建时指向上层产物，
    否则后一层会把前一层的改动覆盖掉（两层都从同一个旧基线出发）。"""
    global BASE
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    tg = json.loads(TARGETS.read_text(encoding="utf-8"))
    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    s8b = json.loads((R / "_semi_qwen3vl_som_all.json").read_text(encoding="utf-8"))
    base = json.loads(SP.find(base_stem).read_text(encoding="utf-8"))
    BASE = base
    inv: dict = {}
    for qid, core in {**tg["target"], **tg["control"]}.items():
        inv.setdefault((q[qid]["visible"], core.lower()), []).append(qid)
    return raw, tg, q, s8b, base, inv


def regress(raw, inv, s8b) -> None:
    """p1 回归：p1 框必须与原 `bbox_all` 一致。

    三类结果必须分开计数，否则会误报：
      完全一致      —— p1 逐位复现原表（绝大部分）
      前缀超集      —— 原表恰是 p1 的前缀 → 说明原全量被 `max_new_tokens=512`
                       截断（实测原表框数上限恰为 15–16，105 条卡在该值），
                       这是**已解释的差异**，不是环境偏差
      真不一致      —— 以上都不是 → 环境/预处理/解析口径有偏差，必须查清
    只有出现"真不一致"才阻断后续对比。
    """
    print("===== 1) p1 回归测试（与原枚举表比对）=====")
    n_exact = n_pre = 0
    real: list = []
    for key, rec in sorted(raw.items()):
        qids = inv.get(tuple(key.split("||")), [])
        orig = next((s8b[i]["bbox_all"] for i in qids
                     if s8b.get(i, {}).get("bbox_all")), None)
        if orig is None:
            continue
        p1 = rec.get("p1") or []
        if len(p1) == len(orig) and all(
                abs(a - b) <= 1e-6 for x, y in zip(orig, p1)
                for a, b in zip(x, y)):
            n_exact += 1
        elif len(p1) > len(orig) and all(
                abs(a - b) <= 1e-6 for x, y in zip(orig, p1[:len(orig)])
                for a, b in zip(x, y)):
            n_pre += 1
        else:
            real.append((key, len(orig), len(p1)))
    tot = n_exact + n_pre + len(real)
    print(f"  可对照 {tot} 对：完全一致 {n_exact} | 原表为 p1 前缀（被 512-token "
          f"截断）{n_pre} | ★真不一致 {len(real)}")
    if real:
        print("  ★ 真不一致样例:", real[:6])
        print("  ⇒ 存在环境差异，后续对比不可用，需先查清")
    else:
        print("  ✓ 无环境偏差（截断类差异已解释）：后续对比成立")


def fidelity(raw, tg, q, s8b, inv, strategy) -> None:
    """控制组保真度：合并表所选框 vs 原表所选框。"""
    print(f"\n===== 2) 控制组保真度（策略 {strategy}）=====")
    st = Counter()
    d_iou = []
    fr = []
    for qid in sorted(tg["control"]):
        core = tg["control"][qid]
        rec = raw.get(f'{q[qid]["visible"]}||{core.lower()}')
        if rec is None:
            st["无重跑数据"] += 1
            continue
        table = merge(rec, strategy)
        fr.append(frag(table))
        orig = (s8b.get(qid) or {}).get("bbox_all") or []
        a, b = pick(q[qid]["query"], orig), pick(q[qid]["query"], table)
        st["原表不可推算" if a is None else "可推算"] += 1
        if a is None or b is None:
            st["合并后不可推算/非法" if b is None else "? "] += 1
            continue
        i = F.iou(a, b)
        d_iou.append(i)
        st[f"{'一致' if i >= F.IOU_KEEP else '不一致'}"
           f"(IoU{'≥.5' if i >= F.IOU_KEEP else '<.5'})"] += 1
        st[f"  精确一致(IoU≥.9)"] += (i >= 0.9)
    for k, v in st.most_common():
        print(f"  {k:24s} {v}")
    if d_iou:
        chg = sum(1 for i in d_iou if i < F.IOU_KEEP)
        print(f"  → 合并后所选框改变 {chg}/{len(d_iou)} = {chg/len(d_iou):.1%}"
              f"（改变率越低越保真）")
    if fr:
        mb, mc = statistics.mean(x[0] for x in fr), statistics.mean(x[1] for x in fr)
        print(f"  碎片化：平均框数 {mb:.1f} → IoU0.3 聚类 {mc:.1f}"
              f"（簇/框 = {mc/mb:.2f}，越接近 1 越干净）")


def target(raw, tg, q, base, strategy, axes_note=True) -> tuple[Counter, list]:
    """目标集改动面：合并表相对当前最优提交。"""
    print(f"\n===== 3) 目标集改动面（策略 {strategy}）=====")
    st = Counter()
    chg = []
    fr = []
    for qid in sorted(tg["target"]):
        core = tg["target"][qid]
        rec = raw.get(f'{q[qid]["visible"]}||{core.lower()}')
        if rec is None:
            st["无重跑数据"] += 1
            continue
        table = merge(rec, strategy)
        fr.append(frag(table))
        st[f"候选数{min(len(table), 8) if len(table) < 8 else '8+'}"] += 1
        new = pick(q[qid]["query"], table)
        if new is None:
            st["仍不可推算/非法 → 不动"] += 1
            continue
        old = base[qid]["bbox"]
        if F.iou(new, old) >= F.IOU_KEEP:
            st["与现答案一致 → 不动"] += 1
            continue
        st["★ 会改变"] += 1
        chg.append((qid, old, new))
    for k, v in st.most_common():
        print(f"  {k:26s} {v}")
    if fr:
        mb, mc = statistics.mean(x[0] for x in fr), statistics.mean(x[1] for x in fr)
        print(f"  碎片化：平均框数 {mb:.1f} → IoU0.3 聚类 {mc:.1f}"
              f"（簇/框 = {mc/mb:.2f}，越接近 1 越干净）")
    print(f"  → 会改变 {len(chg)} 条 | 最大 Δ = {len(chg)/5690:.4f}"
          f" | ×30% 期望 ≈ {len(chg)*0.30/5690:.4f}")
    return st, chg


def build_submission(raw, tg, q, base, strategy, stem):
    """生成新提交件（只改 bbox；其余字段原样回显）。"""
    out = {k: dict(v) for k, v in base.items()}
    n = 0
    for qid in sorted(tg["target"]):
        core = tg["target"][qid]
        rec = raw.get(f'{q[qid]["visible"]}||{core.lower()}')
        if rec is None:
            continue
        new = pick(q[qid]["query"], merge(rec, strategy))
        if new is None or F.iou(new, base[qid]["bbox"]) >= F.IOU_KEEP:
            continue
        out[qid]["bbox"] = new
        n += 1
    _write_zip(out, stem)
    return n


def subset_eval(raw, tg, q, base, build=False, stem=None,
                iou_agree=0.5, frag_min=0.6, n_max=12) -> tuple[list, list]:
    """分层验证的**第一层：跨策略一致的「高置信子集」**。

    判据事先写定（避免事后按结果挑参数）：
      ① 跨策略一致：B/C/D 中凡能给出合法所选框者，两两 IoU ≥ iou_agree
         —— 合并策略换来换去答案都一样，说明这个答案不依赖合并的细节
      ② 碎片率 簇/框 ≥ frag_min（IoU0.3 聚类）——表内没有大量同实例碎片
      ③ 候选数 ≤ n_max（避免超长表把"第 k 个"推得太远）
    所选框取策略 C（①已保证三策略等价，C 召回最全）。

    返回 (入选项列表, 观察项列表)；条目为 (qid, old, new, n, frag_ratio, 原因)。
    """
    picked, held = [], []
    for qid in sorted(tg["target"]):
        core = tg["target"][qid]
        rec = raw.get(f'{q[qid]["visible"]}||{core.lower()}')
        if rec is None:
            continue
        qy = q[qid]["query"]
        per = {s: (merge(rec, s), None) for s in ("B", "C", "D", "E")}
        per = {s: (t, pick(qy, t)) for s, (t, _x) in per.items()}
        avail = {s: b for s, (_t, b) in per.items() if b is not None}
        if len(avail) < 2:
            held.append((qid, "可选策略不足 2 个", None, None, None))
            continue
        bs = list(avail.values())
        worst = min(F.iou(a, b) for i, a in enumerate(bs) for b in bs[i + 1:])
        tab = per["E"][0]
        nb, nc = frag(tab)
        ratio = nc / nb if nb else 0.0
        new = per["E"][1] or per["D"][1] or per["C"][1] or per["B"][1]
        old = base[qid]["bbox"]
        if F.iou(new, old) >= F.IOU_KEEP:
            held.append((qid, "与现答案一致（无改动）", old, new, nb))
            continue
        if worst < iou_agree:
            held.append((qid, f"跨策略不一致(minIoU {worst:.2f})", old, new, nb))
        elif ratio < frag_min:
            held.append((qid, f"碎片率高(簇/框 {ratio:.2f})", old, new, nb))
        elif nb > n_max:
            held.append((qid, f"候选数过大({nb})", old, new, nb))
        else:
            picked.append((qid, old, new, nb, ratio))
    print(f"\n===== 4) 分层第一层：高置信子集 =====")
    print(f"  判据：跨策略 IoU≥{iou_agree} 且 簇/框≥{frag_min} 且 候选数≤{n_max}")
    print(f"  ★ 入选 {len(picked)} 条 | 观察 {len(held)} 条")
    for k, v in Counter(x[1].split("(")[0] for x in held).most_common():
        print(f"    观察原因 {k:28s} {v}")
    if picked:
        print(f"  入选样例：")
        for r in picked[:5]:
            print(f"    {r[0]}  n={r[3]} 簇/框={r[4]:.2f}")
    if build:
        out = {k: dict(v) for k, v in base.items()}
        for qid, _o, new, _n, _r in picked:
            out[qid]["bbox"] = new
        _write_zip(out, stem)
    return picked, held


def trunc_qids(q, cores, coremap, s8b) -> list:
    """截断受影响条目：原表框数 ≥ TRUNC_MIN_N，且排序轴取「远/右/下端」。

    机制：原全量 `max_new_tokens=512` 使枚举输出被砍在 ~15 框（原表框数分布上限
    恰为 15–16）。LTR 取第 k 个时前 15 个仍在，索引不受影响；但 RTL/BTT 是从
    另一端数，被砍掉的正是那端 → 索引必然错位。故只取这两类。
    """
    out = []
    for k in sorted(cores):
        rec = s8b.get(k) or {}
        nc = rec.get("n_cands")
        if nc is None or nc < TRUNC_MIN_N:
            continue
        ax, ori = F.ordinal_axis(q[k]["query"], AXES)
        kk = F.ordinal_index(q[k]["query"])
        if kk is None or F.ordinal_is_reference(q[k]["query"]):
            continue
        if (ax == "x" and ori == "RTL") or (ax == "y" and ori == "BTT"):
            out.append(k)
    return out


def trunc_eval(raw, q, cores, coremap, s8b, base, build=False, stem=None):
    """截断组用**策略 A（p1 单提示，零合并）**——本组唯一需要的修复就是 token 上限。

    不合并的理由是实测的：B/E 虽多改几条，但表长从 17.6 涨到 31.8/27.2，把合并
    噪声也带进来了；而 A 的表就是原提示的完整输出，且已验证其为原表的严格超集、
    前缀逐位相同。故这一档机制最干净、可解释性最强。
    """
    ids = trunc_qids(q, cores, coremap, s8b)
    print(f"\n===== 5) 截断组（策略 A：p1 零合并）=====")
    print(f"  截断受影响条目 {len(ids)} 条")
    cov, chg, fr, strict = [], [], [], 0
    dedup_eff = []
    for k in ids:
        rec = raw.get(f'{q[k]["visible"]}||{coremap[k].lower()}')
        if rec is None:
            continue
        cov.append(k)
        orig = (s8b.get(k) or {}).get("bbox_all") or []
        # 前缀判定必须用**原始 p1**（未排序、未去重）——与 `regress()` 同口径。
        # 用 merge() 的结果比是错的：merge 会按 x 重排并去重，本就不该与原表逐位相等
        # （曾据此误判 24 条"前缀不符"，其中 7 条还出现"完整表更短"——那是去重删掉了
        # 近重复框的假象）。
        p1 = rec.get("p1") or []
        if len(p1) >= len(orig) and all(
                abs(a - b) <= 1e-6 for x, y in zip(orig, p1[:len(orig)])
                for a, b in zip(x, y)):
            strict += 1
        tab = merge(rec, "A")
        dedup_eff.append(len(p1) - len(tab))
        nb, nc = frag(tab)
        fr.append(nc / nb if nb else 0.0)
        new = pick(q[k]["query"], tab)
        if new is None or F.iou(new, base[k]["bbox"]) >= F.IOU_KEEP:
            continue
        chg.append((k, base[k]["bbox"], new, len(orig), len(tab)))
    print(f"  已覆盖 {len(cov)}/{len(ids)} | 原始 p1 为原表严格超集（确证被截断）：{strict}/{len(cov)}")
    if dedup_eff:
        print(f"  去重+排序的影响：平均删掉 {statistics.mean(dedup_eff):.1f} 个近重复框")
    print(f"  ★ 会改变 {len(chg)} 条 | 最大 Δ = {len(chg)/5690:.4f}"
          f" | ×30% 期望 ≈ {len(chg)*0.30/5690:.4f}")
    if fr:
        print(f"  碎片率 {statistics.mean(fr):.2f}")
    for k, o, n, no, nf in chg[:6]:
        print(f"    {k}  原表 {no} 框 → 完整 {nf} 框  "
              f"旧{[round(x, 3) for x in o]} → 新{[round(x, 3) for x in n]}")
    if build:
        out = {k2: dict(v) for k2, v in base.items()}
        for k, _o, n, _no, _nf in chg:
            out[k]["bbox"] = n
        _write_zip(out, stem)
    return chg


def _write_zip(out: dict, stem: str) -> int:
    """写提交件（JSON + zip，固定时间戳保证容器 SHA 稳定）并跑硬约束校验。

    本项目只保留一处打包实现——之前多处各写一份，改格式时会漏改。
    """
    import hashlib
    import subprocess
    import zipfile
    (SP.out_path(stem, ".json")).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    p = SP.out_path(stem, ".zip")
    payload = json.dumps(out, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo(f"{stem}.json", date_time=(1980, 1, 1, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, payload)
    n = sum(1 for k in out if out[k]["bbox"] != BASE.get(k, {}).get("bbox"))
    print(f"\n已写 {p.name}  相对当前最优提交改动 {n} 条  SHA256 "
          f"{hashlib.sha256(p.read_bytes()).hexdigest().upper()[:16]}…")
    r = subprocess.run([sys.executable, "-B", str(_HERE / "_validate_semi.py"), str(p)],
                       capture_output=True, text=True)
    print("  " + "\n  ".join(ln for ln in r.stdout.splitlines()
                              if ln.strip().startswith(("[", "✓", "✗"))))
    return n


def render(raw, tg, q, base, strategy, n_show, which="target") -> None:
    """目检图（颜色含义固定，便于判读）：

        橙  = p1 框（原管线所用提示，可信基线）
        蓝  = 合并新增（来自 p2/p3/分块提示）
        绿粗 = 所选第 k 框（本次会写入提交的框）
        品红 = 现答案（当前已提交件里的框）

    控制组的图同样要出：那里知道**原表所选框**（≈品红），所以能直接判断"合并
    新增的蓝框是真实例、还是同一实例的碎片"——这是选定策略的终判依据。
    """
    from PIL import Image, ImageDraw
    # 目录带策略名：不同策略的渲染必须共存，才能与已有判读对照（曾因同名覆盖丢掉对照组）
    out_dir = VIZ / f"{which}_{strategy}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shown = 0
    for qid in sorted(tg[which]):
        if shown >= n_show:
            break
        core = tg[which][qid]
        rec = raw.get(f'{q[qid]["visible"]}||{core.lower()}')
        if rec is None:
            continue
        table = merge(rec, strategy)
        new = pick(q[qid]["query"], table)
        if new is None:
            continue
        p1 = [b for b in dedup(rec.get("p1") or [])]
        try:
            im = Image.open(SEMI / q[qid]["visible"]).convert("RGB")
        except Exception:
            continue
        im = im.resize((im.width * 2, im.height * 2))
        d = ImageDraw.Draw(im)
        W, H = im.size
        for i, b in enumerate(table, 1):
            in_p1 = any(F.iou(b, o) > DEDUP_IOU for o in p1)
            col = (255, 140, 0) if in_p1 else (0, 120, 255)
            d.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H],
                        outline=col, width=3)
            d.text((b[0] * W + 3, b[1] * H + 3), str(i), fill=col)
        d.rectangle([new[0] * W, new[1] * H, new[2] * W, new[3] * H],
                    outline=(0, 220, 0), width=6)
        d.rectangle([base[qid]["bbox"][0] * W, base[qid]["bbox"][1] * H,
                     base[qid]["bbox"][2] * W, base[qid]["bbox"][3] * H],
                    outline=(255, 0, 200), width=2)
        im.save(out_dir / f"{qid}.png")
        n_add = sum(1 for b in table
                    if not any(F.iou(b, o) > DEDUP_IOU for o in p1))
        print(f"  {qid} n={len(table)}(橙{len(table)-n_add}/蓝{n_add}) "
              f"k={F.ordinal_index(q[qid]['query'])} "
              f"| {q[qid]['query'][:56]}")
        shown += 1
    print(f"已存 {shown} 张（{which}）→ {VIZ}")
    print("  橙=p1原提示框 蓝=合并新增 绿粗=所选第k框 品红=现答案")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="C", choices=list("ABCDE"))
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--fidelity", action="store_true")
    ap.add_argument("--target", action="store_true")
    ap.add_argument("--zip", action="store_true", help="生成提交件")
    ap.add_argument("--render", type=int, default=0)
    ap.add_argument("--render-set", default="target",
                    choices=["target", "control"], help="渲染哪一组（控制组用于判碎片）")
    ap.add_argument("--subset", action="store_true",
                    help="分层第一层：跨策略一致的高置信子集")
    ap.add_argument("--zip-subset", action="store_true",
                    help="把高置信子集打成提交件")
    ap.add_argument("--frag-min", type=float, default=0.6)
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--trunc", action="store_true",
                    help="截断组（策略 A 零合并）：原枚举被 512-token 上限截断的条目")
    ap.add_argument("--zip-trunc", action="store_true", help="把截断组打成提交件")
    ap.add_argument("--base", default="result_semi_ord_det_ss_y_qfb",
                    help="基线提交件名（不含扩展名）；分层构建时指向上层产物")
    ap.add_argument("--stem", default="", help="提交件基名（默认按策略自动命名）")
    a = ap.parse_args()
    raw, tg, q, s8b, base, inv = load(a.base)
    print(f"原始框文件 {RAW.name}：{len(raw)} 对 | 目标 {len(tg['target'])} 条 "
          f"| 控制 {len(tg['control'])} 条")
    if a.regress:
        regress(raw, inv, s8b)
    if a.fidelity:
        # 五种策略全测：保真度是选策略的依据，必须同口径横向比较
        for s in ("A", "B", "C", "D", "E"):
            fidelity(raw, tg, q, s8b, inv, s)
    if a.target:
        target(raw, tg, q, base, a.strategy)
    if a.zip:
        build_submission(raw, tg, q, base, a.strategy,
                         a.stem or f"result_semi_enum_{a.strategy}")
    if a.subset or a.zip_subset:
        subset_eval(raw, tg, q, base, build=a.zip_subset,
                    stem="result_semi_enum_subset1",
                    frag_min=a.frag_min, n_max=a.n_max)
    if a.trunc or a.zip_trunc:
        cores = set(json.loads((R / "_ordinal_cores_r2.json").read_text(
            encoding="utf-8")))
        coremap = json.loads((R / "_ordinal_cores_r2.json").read_text(
            encoding="utf-8"))
        trunc_eval(raw, q, cores, coremap, s8b, base, build=a.zip_trunc,
                   stem="result_semi_enum_trunc")
    if a.render:
        render(raw, tg, q, base, a.strategy, a.render, a.render_set)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
