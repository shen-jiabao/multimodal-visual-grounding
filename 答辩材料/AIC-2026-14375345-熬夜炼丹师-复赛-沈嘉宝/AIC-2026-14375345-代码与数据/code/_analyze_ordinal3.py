# -*- coding: utf-8 -*-
"""核心词抽取 v3：处理 "Nth closest"（序数词在名词之后）型。

- 序数词之后有名词 → 取其后头名词短语
- 否则 → 取序数词之前的尾名词短语（英式 NP 头名词在尾部）

v3.1 改动（2026-09-21）：参数化数据集路径，供初赛/复赛共用**同一套判据**。
判据（`ORD`/`CUT`/`TRAIL_JUNK`/`ART`/`MAXW`/`head_np`/`tail_np`/`extract_head`）
一字未改；重构后须验证初赛产物逐字节不变。

用法：
    python _analyze_ordinal3.py                          # 默认初赛 → _ordinal_cores_v3.json
    python _analyze_ordinal3.py --dataset 复赛数据集-... --out 复赛/_ordinal_cores_v3.json
"""
import argparse
import json
import os
import re
from collections import defaultdict
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

R = ROOT_S

ORD = re.compile(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|"
                 r"ninth|tenth|last)\b", re.I)
CUT = re.compile(
    r"\s*[,;]\s*|\b(located|which|that|who|standing|sitting|walking|seen|"
    r"positioned|visible|in front of|behind|next to|beside|near|"
    r"on the|at the|to the|inside|outside|under|above|below|"
    r"with|holding|wearing|facing|leaning|riding|carrying|parked|"
    r"closest|farthest|nearest|furthest|from|between)\b", re.I)
TRAIL_JUNK = re.compile(
    r"\b(standing|sitting|walking|located|positioned|visible|seen|facing|"
    r"of|the|a|an|and|with|on|in|at|to|from|beside|next|its|his|her|"
    r"their|neck|legs|leg|arm|arms|head|tail|mouth|eye|eyes|paw|paws)\b$",
    re.I)
ART = re.compile(r"^\s*(the|a|an|its|his|her|their|this|that)\s+", re.I)
MAXW = 5


def head_np(seg: str) -> str:
    seg = ART.sub("", seg).strip()
    seg = CUT.split(seg)[0].strip()
    seg = re.sub(r"[^\w\s\-']", " ", seg)
    return " ".join(re.sub(r"\s+", " ", seg).split()[:MAXW]).strip()


def tail_np(seg: str) -> str:
    """取段尾的头名词（反复剥离尾部虚词/动词/身体部位修饰）。"""
    seg = re.sub(r"[^\w\s\-']", " ", seg)
    words = seg.split()
    for _ in range(6):
        while words and TRAIL_JUNK.match(words[-1]):
            words.pop()
        if not words:
            return ""
        break
    # 取最后 ≤3 个实词
    return " ".join(words[-3:]).strip()


def extract_head(q: str) -> str:
    q = re.sub(r"\([^)]*\)", " ", q)
    m = ORD.search(q)
    if m:
        after = head_np(q[m.end():])
        if len(after) >= 3:
            return after
        before = tail_np(q[:m.start()])
        if len(before) >= 3:
            return before
        # 兜底：整句头名词
        return head_np(q)
    return head_np(q)


def cleanup(c: str) -> str:
    """核心词清理（**判据唯一实现处**，由 `_launch_qwen_full` 转发引用）。

    这是第二段处理：`extract_head` 之后再做一次规整。
    历史教训：`cleanup` 原先只存在于 `_launch_qwen_full.py`，导致
    `_analyze_ordinal3.py` 单独跑出的核心词与线上实际使用的 **38 条不同**
    （如 `rectangular panel mounted` vs 线上 `rectangular panel`）。
    故统一到此处，并已用"初赛产物逐字节一致"验证（SHA 531E059D…）。
    """
    c = re.sub(r"^(and|or)\s+", "", c, flags=re.I)
    c = re.sub(r"^(the|a|an)\s+", "", c, flags=re.I)
    c = re.sub(r"\s+(facing|mounted|wrapped|standing|lowered|held|attached|"
               r"placed|hung|hanging|printed|drawn|leaning|sitting|walking)\b.*$",
               "", c, flags=re.I)
    c = re.sub(r"\s+of\s+the\s*$", "", c, flags=re.I)
    c = re.sub(r"[\s\-]+$", "", c)
    return c.strip()


def build_cores(queries: dict, clean: bool = True) -> dict:
    """对全部序数问句抽取核心词，返回 {qid: core}。

    `clean=True` 时套用 `cleanup`，与线上实际使用的核心词完全一致。
    """
    raw = {k: extract_head(v["query"])
           for k, v in queries.items() if ORD.search(v["query"])}
    return {k: cleanup(v) for k, v in raw.items()} if clean else raw


def report(cores: dict, queries: dict) -> None:
    """打印失败率、修复样例、SoM 调用量与耗时估算。"""
    bad = [k for k, c in cores.items() if len(c) < 3]
    print(f"序数问句 {len(cores)}；仍无法抽取 {len(bad)} "
          f"({100*len(bad)/len(cores):.1f}%)")
    print("\n=== 仍失败样例 ===")
    for k in bad[:12]:
        print(f"  [{cores[k]}]  <-  {queries[k]['query'][:80]}")
    print("\n=== 'Nth closest' 型修复样例 ===")
    for k in list(cores):
        if re.search(r"\b(second|third|fourth)\b.*\b(closest|from the)\b",
                     queries[k]["query"], re.I):
            print(f"  [{cores[k]}]  <-  {queries[k]['query'][:76]}")
    groups = defaultdict(set)
    for k, c in cores.items():
        groups[(os.path.basename(queries[k]["visible"]), c.lower())].add(k)
    print(f"\n唯一 (图, 核心词) 组: {len(groups)}")
    print(f"SoM 总调用 = {len(groups)} + {len(cores)} = "
          f"{len(groups)+len(cores)}；≈{(len(groups)+len(cores))*3/3600:.1f}h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="初赛数据集-基于大模型的多模态视觉理解与推理")
    ap.add_argument("--out", default="_ordinal_cores_v3.json")
    a = ap.parse_args()

    qp = os.path.join(R, a.dataset, "queries", "queries.json")
    queries = json.load(open(qp, encoding="utf-8"))
    cores = build_cores(queries)
    report(cores, queries)

    out = a.out if os.path.isabs(a.out) else os.path.join(R, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(cores, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"已存 {os.path.relpath(out, R)}")


if __name__ == "__main__":
    main()
