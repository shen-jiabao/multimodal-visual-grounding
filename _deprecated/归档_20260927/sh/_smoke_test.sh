#!/bin/bash
# 端到端冒烟测试：用**生产脚本本身**跑一个 12 图 / 双阶段的小数据集。
# 目的：验证整条接线（路径解析、核心词装配、两阶段、检查点、输出 schema），
# 不污染正式产物——数据用符号链接，输出走 QWEN_OUT_* 覆盖到 /srv/smoke/。
set -e
PY=/root/miniconda3/envs/myconda/bin/python
SM=/srv/smoke
rm -rf "$SM"
mkdir -p "$SM/dataset/Images/visible" "$SM/dataset/queries" "$SM/out"

echo '===== 1) 构造临时数据集（符号链接 + 过滤 queries）====='
$PY - <<'PY'
import json, os
D, SM = "/srv/dataset", "/srv/smoke"
q = json.load(open(os.path.join(D, "queries", "queries.json"), encoding="utf-8"))
cores = json.load(open("/srv/output/_ordinal_cores_v3.json", encoding="utf-8"))
have = set(os.listdir(os.path.join(D, "Images", "visible")))

from collections import defaultdict
byimg = defaultdict(list)
for k, v in q.items():
    if os.path.basename(v["visible"]) in have:
        byimg[v["visible"]].append(k)

# 选 12 张：必须同时含 direct 与序数问句，才能同时压到两个阶段
picked = []
for img, ks in sorted(byimg.items()):
    has_d = any(k not in cores for k in ks)
    has_o = any(k in cores for k in ks)
    if has_d and has_o:
        picked.append(img)
    if len(picked) >= 12:
        break
sub = {k: q[k] for img in picked for k in byimg[img]}
print(f"  选中图 {len(picked)} 张 | 问句 {len(sub)} 条 "
      f"(direct {sum(1 for k in sub if k not in cores)} / "
      f"序数 {sum(1 for k in sub if k in cores)})")
for img in picked:
    os.symlink(os.path.join(D, img),
               os.path.join(SM, "dataset", img))
json.dump(sub, open(os.path.join(SM, "dataset", "queries", "queries.json"),
                    "w", encoding="utf-8"), ensure_ascii=False)
print("  已建符号链接与过滤 queries.json")
PY

echo
echo '===== 2) 运行生产管线（路径经环境变量覆盖）====='
cd /root
QWEN_DATA="$SM/dataset" \
QWEN_OUT_DIRECT="$SM/out/qwen3vl_all.json" \
QWEN_OUT_SOM="$SM/out/qwen3vl_som_all.json" \
QWEN_CORES="/srv/output/_ordinal_cores_v3.json" \
$PY -u _qwen_full_pipeline.py 2>&1 | tail -n 25

echo
echo '===== 3) 输出校验 ====='
$PY - <<'PY'
import json, os
SM = "/srv/smoke/out"
qd = json.load(open("/srv/smoke/dataset/queries/queries.json", encoding="utf-8"))
cores = json.load(open("/srv/output/_ordinal_cores_v3.json", encoding="utf-8"))
d = json.load(open(os.path.join(SM, "qwen3vl_all.json"), encoding="utf-8"))
s = json.load(open(os.path.join(SM, "qwen3vl_som_all.json"), encoding="utf-8"))
print(f"  direct 条数 {len(d)} / 应 {len(qd)} | 键集相等: {set(d)==set(qd)}")
nb = sum(1 for v in d.values() if v.get("bbox"))
err = sum(1 for v in d.values() if v.get("err"))
print(f"  direct 出框 {nb} | err {err}")
ordi = {k for k in qd if k in cores}
print(f"  som 条数 {len(s)} / 应 {len(ordi)} | 键集相等: {set(s)==set(ordi)}")
ns = sum(1 for v in s.values() if v.get("bbox"))
nc = sum(1 for v in s.values() if (v.get('n_cands') or 0) > 1)
print(f"  som 出框 {ns} | 枚举>1候选 {nc}")
for k in list(s)[:3]:
    v = s[k]
    print(f"    样例 {k}: core={v.get('core')!r} n_cands={v.get('n_cands')} "
          f"num={v.get('num')} bbox={v.get('bbox')}")
for k in list(d)[:2]:
    print(f"    样例 {k}: n={d[k].get('n')} bbox={d[k].get('bbox')}")
# 字段契约：融合器要读 bbox/n_cands/num，缺一不可
miss = [k for k, v in s.items() if not {"core", "n_cands", "bbox"} <= set(v)]
print(f"  som 缺字段条目: {len(miss)}")
PY
echo
echo '===== 冒烟测试结束 ====='
