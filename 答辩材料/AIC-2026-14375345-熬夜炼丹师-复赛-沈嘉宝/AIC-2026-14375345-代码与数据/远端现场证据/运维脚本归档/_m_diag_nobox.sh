#!/bin/bash
# 诊断 M 件全量的 329 条无框：是"请求失败"、"解析不到"、还是"截断"？
set -u
R=/root/rivermind-data
$R/venv_q38/bin/python - "$R/output/m_direct.json" <<'PY'
import collections
import json
import sys

d = json.load(open(sys.argv[1], encoding="utf-8"))
nob = {k: v for k, v in d.items() if not v.get("bbox")}
print(f"总 {len(d)}｜有框 {len(d)-len(nob)}｜无框 {len(nob)}"
      f"（{len(nob)/len(d)*100:.1f}%）")

err = {k: v for k, v in nob.items() if v.get("err")}
noerr = {k: v for k, v in nob.items() if not v.get("err")}
print(f"  其中带 err(请求级失败): {len(err)}｜无 err(解析不到): {len(noerr)}")

print("\n--- ① err 类型分布 ---")
for msg, c in collections.Counter(
        v["err"][:60] for v in err.values()).most_common(10):
    print(f"  {c:>4}  {msg}")

print("\n--- ② 无 err 的 raw 形态分布 ---")
pat = collections.Counter()
for v in noerr.values():
    r = (v.get("raw") or "")
    if not r:
        pat["raw 为空"] += 1
    elif "null" in r:
        pat["raw 含 null（模型判不可见）"] += 1
    elif r.count("[") == 0:
        pat["raw 无 '['（非数组）"] += 1
    elif r.count("[") > r.count("]"):
        pat["raw 括号不闭合（★截断）"] += 1
    else:
        pat["raw 有形但该条 idx 缺"] += 1
for k, c in pat.most_common():
    print(f"  {c:>4}  {k}")

print("\n--- ③ 无框是否集中在合并批次的靠后位置（截断特征）---")
cnt = collections.Counter()
tot = collections.Counter()
for k, v in d.items():
    if v.get("err") or "i" not in v:
        continue
    tot[v["i"]] += 1
    if not v.get("bbox"):
        cnt[v["i"]] += 1
print("   批内序号 i : 无框/总数（占比）")
for i in sorted(tot):
    print(f"     {i:>3} : {cnt[i]:>4}/{tot[i]:<5} "
          f"({cnt[i]/tot[i]*100:5.1f}%)")

print("\n--- ④ 每图 query 数 vs 无框率 ---")
by = collections.defaultdict(lambda: [0, 0])
for k, v in d.items():
    if v.get("err") or "i" not in v:
        continue
    n = v.get("_n") or None
    by[0][1] += 1   # 占位
grp = collections.defaultdict(lambda: [0, 0])
img = collections.defaultdict(list)
for k, v in d.items():
    if v.get("err") or "i" not in v:
        continue
    img[k.rsplit("_", 1)[0]].append(v)
for q, vs in img.items():
    n = max(v["i"] for v in vs)
    grp[n][0] += sum(1 for v in vs if v.get("bbox"))
    grp[n][1] += len(vs)
print("   同图 query 数 : 有框/总数（无框率）")
for n in sorted(grp):
    ok, t = grp[n]
    print(f"     {n:>3} : {ok:>5}/{t:<6} ({(t-ok)/t*100:5.1f}%)")

print("\n--- ⑤ 抽样 5 条无 err 的 raw ---")
for k in list(noerr)[:5]:
    print(f"  {k} raw={ (noerr[k].get('raw') or '')[:150]!r}")
print("\n--- ⑥ 抽样 3 条带 err 的 ---")
for k in list(err)[:3]:
    print(f"  {k} err={err[k]['err'][:140]!r}")
PY
echo "=== DONE ==="
