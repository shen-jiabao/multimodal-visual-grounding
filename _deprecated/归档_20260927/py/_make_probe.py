# -*- coding: utf-8 -*-
"""生成 mini 探针包：选 5 张覆盖不同场景的 visible 图 + 对应 queries 子集。

场景：多目标(000023 石墩)、密集(14问句图)、单目标、小目标、遮挡。
输出到 d:\Desktop\2026AiC\probe\ （Images/visible + mini_queries.json + mini_test.json）
"""
import json
import os
import re
import shutil
from pathlib import Path

R = Path(r"d:\Desktop\2026AiC")
DATA = R / "初赛数据集-基于大模型的多模态视觉理解与推理"
Q = json.loads((DATA / "queries" / "queries.json").read_text(encoding="utf-8"))
TEST = json.loads((DATA / "test.json").read_text(encoding="utf-8"))

PROBE_DIR = R / "probe"
VIS_OUT = PROBE_DIR / "Images" / "visible"
VIS_OUT.mkdir(parents=True, exist_ok=True)

# 选图策略：程序化选 5 张覆盖不同难度
img_qcount = {}
count_words = re.compile(r"\b(all|every|couple|pair|two|three|four|several|group)\b", re.I)
size_words = re.compile(r"\b(small|little|tiny|fit)\b", re.I)
occ_words = re.compile(r"\b(occlud|behind|behind |under |inside |partially)\b", re.I)
by_img = {}
for k, v in Q.items():
    iid = os.path.splitext(os.path.basename(v["visible"]))[0]
    by_img.setdefault(iid, []).append(k)

picks = []
# 1. 相位数序数多目标（000023 已知验过的图）
picks.append("000023")
# 2. 密集（问句最多的图）
img_cnt = sorted(by_img.items(), key=lambda x: -len(x[1]))
picks.append(img_cnt[0][0])  # 001038 (14 问句)
# 3. 单目标简单（句子短、无复数词/序数句）
simple = []
for k, v in Q.items():
    ql = v["query"].lower()
    if len(v["query"]) < 50 and not count_words.search(ql) \
            and not re.search(r"\b(first|second|third|fourth)\b", ql) \
            and not size_words.search(ql) and not occ_words.search(ql):
        simple.append(k)
    if len(simple) >= 3:
        break
if simple:
    picks.append(os.path.splitext(os.path.basename(Q[simple[0]]["visible"]))[0])
# 4. 小目标：含 small/little/tiny 的 ask 句
small = [k for k in Q if size_words.search(Q[k]["query"].lower())]
if small:
    picks.append(os.path.splitext(os.path.basename(Q[small[0]]["visible"]))[0])
# 5. 遮挡：含 occlud/behind 等
occ = [k for k in Q if occ_words.search(Q[k]["query"].lower())]
if occ:
    picks.append(os.path.splitext(os.path.basename(Q[occ[0]]["visible"]))[0])

# 去重、保持顺序
seen, uniq = set(), []
for p in picks:
    if p not in seen:
        seen.add(p)
        uniq.append(p)
picks = uniq[:5]

mini_q, mini_t = {}, {}
for iid in picks:
    imgname = f"{iid}.png"
    src = DATA / "Images" / "visible" / imgname
    if not src.exists():
        print(f"[SKIP] 无图 {src}")
        continue
    shutil.copy(src, VIS_OUT / imgname)
    for k in by_img[iid]:
        mini_q[k] = Q[k]
        mini_t[k] = TEST[k]

(PROBE_DIR / "mini_queries.json").write_text(
    json.dumps(mini_q, ensure_ascii=False), encoding="utf-8")
(PROBE_DIR / "mini_test.json").write_text(
    json.dumps(mini_t, ensure_ascii=False), encoding="utf-8")

print("选中图:", picks)
print("问句数:", len(mini_q))
for k, v in mini_q.items():
    print("  ", k, "|", v["query"][:80])
print("探针包 ->", PROBE_DIR)