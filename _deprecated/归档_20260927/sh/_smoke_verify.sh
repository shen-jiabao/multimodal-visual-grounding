#!/bin/bash
# 校验冒烟测试产物：键集完整性、字段契约、两阶段是否都产出。
PY=/root/miniconda3/envs/myconda/bin/python
$PY - <<'PY'
import json, os
SM = "/srv/smoke/out"
qd = json.load(open("/srv/smoke/dataset/queries/queries.json", encoding="utf-8"))
cores = json.load(open("/srv/output/_ordinal_cores_v3.json", encoding="utf-8"))
d = json.load(open(os.path.join(SM, "qwen3vl_all.json"), encoding="utf-8"))
s = json.load(open(os.path.join(SM, "qwen3vl_som_all.json"), encoding="utf-8"))

print(f"  临时 queries 条数 {len(qd)}（direct {sum(1 for k in qd if k not in cores)}"
      f" / 序数 {sum(1 for k in qd if k in cores)}）")
print(f"[A] direct 产物 {len(d)} 条 | 键集 == queries: {set(d) == set(qd)}")
print(f"    出框 {sum(1 for v in d.values() if v.get('bbox'))} | "
      f"err {sum(1 for v in d.values() if v.get('err'))}")
ordi = {k for k in qd if k in cores}
print(f"[B] som 产物 {len(s)} 条 | 键集 == 序数集: {set(s) == set(ordi)}")
print(f"    出框 {sum(1 for v in s.values() if v.get('bbox'))} | "
      f"枚举>1候选 {sum(1 for v in s.values() if (v.get('n_cands') or 0) > 1)}")
miss = [k for k, v in s.items() if not {"core", "n_cands", "bbox"} <= set(v)]
print(f"    som 缺必需字段的条目: {len(miss)}")
print()
print("  样例（序数）:")
for k in list(s)[:4]:
    v = s[k]
    print(f"    {k}: core={v.get('core')!r} n_cands={v.get('n_cands')} "
          f"num={v.get('num')} bbox={v.get('bbox')}")
print("  样例（direct）:")
for k in list(d)[:3]:
    print(f"    {k}: n={d[k].get('n')} bbox={d[k].get('bbox')}")
PY
