#!/bin/bash
# 复赛 LocateAnything-3B 探针：构造小样数据集（符号链接，不复制图片）+ 跑探针。
# 解释器可用 PROBE_PY 覆盖（默认隔离 venv），便于对 transformers 版本做 A/B。
set -u
PY=/root/miniconda3/envs/myconda/bin/python
# 可切换解释器：默认隔离 venv，但 venv 的 transformers 4.57.1 实测
# `AutoProcessor` 无法导入（源文件在、懒加载列表缺名）；myconda 的 4.57.6
# 能导入。故允许用 PROBE_PY 覆盖，好做 A/B 对照而不是盲信某一侧。
VPY=${PROBE_PY:-/srv/venv-locate/bin/python}
P=/srv/probe_locate
rm -rf $P
mkdir -p $P/dataset/Images/visible $P/out

echo "== 1) 构造小样（12 图，direct 与序数混合）=="
$PY - <<'PY'
import json, os
from collections import defaultdict
D, P = "/srv/dataset", "/srv/probe_locate"
q = json.load(open(os.path.join(D, "queries", "queries.json"), encoding="utf-8"))
cores = set(json.load(open("/srv/output/_ordinal_cores_v3.json", encoding="utf-8")))
have = set(os.listdir(os.path.join(D, "Images", "visible")))
byimg = defaultdict(list)
for k, v in q.items():
    if os.path.basename(v["visible"]) in have:
        byimg[v["visible"]].append(k)
picked = [i for i, ks in sorted(byimg.items())
          if any(k not in cores for k in ks) and any(k in cores for k in ks)][:12]
sub, nd, no = {}, 0, 0
for img in picked:
    for k in byimg[img]:
        sub[k] = q[k]
        if k in cores: no += 1
        else: nd += 1
    os.symlink(os.path.join(D, img), os.path.join(P, "dataset", img))
json.dump(sub, open(os.path.join(P, "dataset", "mini_queries.json"), "w",
                    encoding="utf-8"), ensure_ascii=False)
print(f"  图 {len(picked)} | 问句 {len(sub)} (direct {nd} / 序数 {no})")
PY

echo
echo "== 2) 跑探针（隔离 venv）=="
PROBE_DATA=$P/dataset \
PROBE_QUERIES=$P/dataset/mini_queries.json \
PROBE_OUT=$P/out/probe_result.json \
  $VPY -u /root/_probe_locale.py 2>&1 | grep -vE '^\s*$' | tail -30

echo
echo "== 3) 显存 =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
