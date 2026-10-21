# -*- coding: utf-8 -*-
"""非序数（direct）答案的离线机械审计——**不需要真值**，用问句自带的可检验约束找"必然错"。

动因：非序数 4587 条是最大盲区（若准确率 85% 就有 688 条错，量级远超序数剩余空间），
而全局无任何真值（初赛 test.json 与 queries.json 同构，无 bbox）。故先用不依赖真值的
约束检验，把"错多少条"从猜变成**下界测量**，并产出可执行的修补名单。

三类检验（都刻意偏保守：**假阳性会误导投入，假阴性只是少发现**）
  1. 颜色不符：问句指定了颜色，而当前框内该颜色像素占比≈0
     —— 复用 `predict_gd_3modal._color_mask` / `box_color_score`，不另写一份
  2. 方位不符：问句给出**无参照物**的绝对方位（left/right/top/bottom/center），
     而框中心落在明显相反的一侧；带参照物的（"to the left of …"）一律跳过
  3. 形态异常：面积过大（疑合并多目标）/ 过小 / 长宽比极端

注意：这些检验只能给出**错题数的下界**，且不能反推准确率——准确率要用归零法提交测量
（见回复中的 M1/M2）。

用法
    python -B _audit_nord.py                # 全量非序数审计
    python -B _audit_nord.py --limit 500    # 抽样（快速看分布）
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
sys.path.insert(0, str(R))

import numpy as np  # noqa: E402

import _fuse_qwen as F  # noqa: E402
import _sub_path as SP  # noqa: E402
import predict_gd_3modal as G  # noqa: E402

SEMI = R / "复赛数据集-基于大模型的多模态视觉理解与推理"
CORES = R / "_ordinal_cores_r2.json"

# 绝对方位（无参照物）。带参照物的句式由 REF_GUARD 排除。
POS = {
    "left": re.compile(r"\b(on the left|left\s+side|in the left|left\s+part|"
                       r"leftmost|left-most|on the leftmost)\b", re.I),
    "right": re.compile(r"\b(on the right|right\s+side|in the right|right\s+part|"
                        r"rightmost|right-most)\b", re.I),
    "top": re.compile(r"\b(at the top|top\s+side|upper|topmost|top-most)\b", re.I),
    "bottom": re.compile(r"\b(at the bottom|bottom\s+side|lower|bottommost|"
                         r"bottom-most)\b", re.I),
    "center": re.compile(r"\b(in the (middle|center)|centered|centre)\b", re.I),
}
REF_GUARD = re.compile(
    r"to\s+the\s+(left|right)\s+of|(left|right)\s+of\s+the|"
    r"side\s+of\s+the|next\s+to|beside|"
    r"between|above\s+the|below\s+the|under\s+the|behind\s+the|in\s+front\s+of|"
    r"on\s+top\s+of|near\s+the|adjacent\s+to|"
    # 最高级形态天然是相对的（"the leftmost wheel on the cart" = 相对推车）→ 一律跳过
    r"leftmost|rightmost|topmost|bottommost|lowermost|uppermost", re.I)
# 颜色检验的严格化：只在「颜色词直接修饰被框物体」时才判。
# 若颜色词之后出现连接词/属格/动词分词，颜色多半修饰别的名词或物体的一部分
# （"the trunk of the green tree"、"a person wearing a white shirt"）→ 跳过。
COLOR_LINKER = re.compile(
    r"\b(wearing|holding|carrying|riding|with|of\s+the|on\s+the|in\s+the|"
    r"under\s+the|next\s+to|behind\s+the)\b", re.I)
BIG_AREA, TINY_AREA, MAX_ASPECT = 0.50, 5e-4, 8.0
MIN_BOX_FOR_COLOR = 0.002     # 太小的框算颜色占比噪声大，不据此判错
COLOR_MISS = 0.02             # 框内目标颜色占比低于此值即判"颜色不符"
COLOR_MIN_AREA = 0.02         # 小目标的颜色占比波动大，只对大框判


def color_is_head(query: str, color: str) -> bool:
    """颜色词是否直接在修饰被框物体（严格口径）。

    规则：颜色词必须出现在**最后一个名词短语**之前且其后不再出现连接词/属格。
    实测反例（旧口径会误判）：'the thick trunk of the main large green tree'、
    'a person wearing a white shirt' —— 前者 green 修饰树、后者框是人是整体。
    """
    m = re.search(rf"\b{re.escape(color)}\b", query, re.I)
    if not m:
        return False
    return not COLOR_LINKER.search(query[m.end():])


def parse_pos(query: str) -> set:
    """问句给出的绝对方位集合；含参照物句式的整条跳过（返回空集）。"""
    if not query or REF_GUARD.search(query):
        return set()
    return {k for k, rx in POS.items() if rx.search(query)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="result_semi_enum_trunc")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    q = json.loads((SEMI / "queries" / "queries.json").read_text(encoding="utf-8"))
    ordinal = set(json.loads(CORES.read_text(encoding="utf-8")))
    base = json.loads(SP.find(a.base).read_text(encoding="utf-8"))
    ids = [k for k in sorted(base) if k not in ordinal]
    if a.limit:
        random.seed(7)
        ids = random.sample(ids, min(a.limit, len(ids)))
    print(f"非序数条目 {len(ids)} 条（总 {len(base)} − 序数 {len(ordinal)}）| 基线 {a.base}")

    st = Counter()
    color_rows, pos_rows, path_rows = [], [], []
    img_cache: dict = {}
    for i, k in enumerate(ids, 1):
        box = base[k]["bbox"]
        qy = q[k]["query"]
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        aw, ah = x2 - x1, y2 - y1
        aspect = max(aw, ah) / max(min(aw, ah), 1e-9)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # 1) 形态异常（仅作分布参考：细长杆/极小目标本就合法，不单独当错）
        if area > BIG_AREA:
            st["形态:面积过大(>0.5)"] += 1
            path_rows.append((k, "面积过大", area, qy))
        elif area < TINY_AREA:
            st["形态:面积过小(<5e-4)"] += 1
        elif aspect > MAX_ASPECT:
            st["形态:长宽比极端(>8)"] += 1
        # 2) 方位不符
        ps = parse_pos(qy)
        if ps:
            st["有绝对方位的条目"] += 1
        for p in ps:
            bad = ((p == "left" and cx >= 0.62) or (p == "right" and cx <= 0.38)
                   or (p == "top" and cy >= 0.62) or (p == "bottom" and cy <= 0.38)
                   or (p == "center" and not (0.15 <= cx <= 0.85)))
            if bad:
                st[f"方位不符:{p}"] += 1
                pos_rows.append((k, p, round(cx, 3), round(cy, 3), qy))
        # 3) 颜色不符（严格口径：颜色词须直接修饰被框物体，且框足够大）
        col = G.parse_target_color(qy)
        if col:
            st["有颜色词的条目"] += 1
            if color_is_head(qy, col) and area >= COLOR_MIN_AREA:
                st["颜色检验可用(严格口径)"] += 1
                rel = q[k]["visible"]
                if rel not in img_cache:
                    img_cache[rel] = G._imread_unicode(str(SEMI / rel))
                img = img_cache[rel]
                if img is not None:
                    rgb = img[:, :, ::-1].copy()
                    sc = G.box_color_score(box, rgb, col)
                    if sc < COLOR_MISS:
                        st[f"颜色不符:{col}"] += 1
                        color_rows.append((k, col, round(sc, 4), qy))
        if i % 500 == 0:
            print(f"  …{i}/{len(ids)}", flush=True)

    print("\n===== 汇总 =====")
    for name, v in sorted(st.items()):
        print(f"  {name:22s} {v}")
    n_bad = len({r[0] for r in color_rows} | {r[0] for r in pos_rows}
                | {r[0] for r in path_rows})
    print(f"\n★ 至少命中一项硬约束的条目：{n_bad} / {len(ids)}"
          f" = {n_bad/len(ids):.1%}（这是**错题数的下界**）")
    for tag, rows, cols in (("颜色不符", color_rows, ("qid", "color", "score", "query")),
                            ("方位不符", pos_rows, ("qid", "pos", "cx", "cy", "query")),
                            ("形态异常", path_rows, ("qid", "kind", "val", "query"))):
        print(f"\n--- {tag} 样例（共 {len(rows)}）---")
        for r in rows[:8]:
            print("   " + "  ".join(str(x)[:56] for x in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
