#!/bin/bash
# 二次诊断：无框的真因是"输出被 max_new 截断"还是"模型真的漏输出/判不可见"？
# 说明：raw 只存了前 200 字符（_direct_multi 的 txt[:200]），故不能靠它看闭合。
# 改为查相关性：无框 ↔ 该批次各 query 文本长度之和。
set -u
R=/root/rivermind-data
$R/venv_q38/bin/python - "$R" <<'PY'
import collections
import json
import os
import re
import sys

R = sys.argv[1]
d = json.load(open(os.path.join(R, "output", "m_direct.json"),
                   encoding="utf-8"))
q = json.load(open(os.path.join(R, "dataset", "queries", "queries.json"),
                   encoding="utf-8"))

# 按图聚合，还原批内顺序（i 字段）
img = collections.defaultdict(list)
for k, v in d.items():
    if "i" not in v:
        continue
    img[k.rsplit("_", 1)[0]].append((v["i"], k, v))

print("=== ① 无框条目的 query 长度 vs 有框条目 ===")


def qlen(k):
    return len(q[k]["query"]) if k in q else 0


nb = [k for k, v in d.items() if not v.get("bbox") and "i" in v]
ok = [k for k, v in d.items() if v.get("bbox") and "i" in v]
for name, ks in (("无框", nb), ("有框", ok)):
    L = sorted(qlen(k) for k in ks)
    if L:
        print(f"  {name}: n={len(L)} 中位长度 {L[len(L)//2]} "
              f"均值 {sum(L)/len(L):.1f} 最长 {L[-1]}")

print("\n=== ② 按「批次总字符数」分档看无框率（截断假说的强检验）===")
rows = []
for im, vs in img.items():
    vs.sort()
    tot = sum(qlen(k) for _, k, _ in vs)
    bad = sum(1 for _, _, v in vs if not v.get("bbox"))
    rows.append((tot, bad, len(vs)))
rows.sort()
bins = [(0, 100), (100, 150), (150, 200), (200, 300), (300, 400),
        (400, 600), (600, 99999)]
print("   批次总字符数区间 : 无框/总数（无框率）")
for lo, hi in bins:
    sel = [r for r in rows if lo <= r[0] < hi]
    if not sel:
        continue
    b = sum(r[1] for r in sel)
    t = sum(r[2] for r in sel)
    print(f"     [{lo:>4},{hi if hi<99999 else '∞':>5}) : {b:>4}/{t:<5} "
          f"({b/t*100:5.1f}%)")

print("\n=== ③ 批内位置 × 批次大小（看尾部是否更易无框）===")
tail = collections.Counter()
tailtot = collections.Counter()
for im, vs in img.items():
    n = len(vs)
    for pos, (i, k, v) in enumerate(vs):
        if pos >= n - 2:            # 最后两条
            tailtot["末两条"] += 1
            if not v.get("bbox"):
                tail["末两条"] += 1
        else:
            tailtot["其余"] += 1
            if not v.get("bbox"):
                tail["其余"] += 1
for kk in ("末两条", "其余"):
    print(f"    {kk}: {tail[kk]}/{tailtot[kk]} "
          f"({tail[kk]/max(1,tailtot[kk])*100:.1f}%)")

print("\n=== ④ 无框样本的 query 全文（前 8 条，看是否有共性）===")
for k in nb[:8]:
    print(f"  {k}  len={qlen(k):>4}  {q[k]['query'][:110]!r}")

print("\n=== ⑤ 有框/无框 的 label 里 'idx' 出现个数（200 字符内）===")
c = collections.Counter()
for k, v in d.items():
    if "i" not in v:
        continue
    nidx = len(re.findall(r'"idx"\s*:', v.get("raw") or ""))
    c[(bool(v.get("bbox")), nidx)] += 1
for (hasbox, nidx), n in sorted(c.items()):
    print(f"  {'有框' if hasbox else '无框'}  raw 内 idx 数={nidx} : {n}")
PY
echo "=== DONE ==="
