#!/bin/bash
# 盘点可用于"多模型融合"的离线直接定位产物
set -u
R=/root/rivermind-data

echo "=== output/ 全量 ==="
ls -la $R/output 2>/dev/null | sed 's/^/  /'

echo; echo "=== 各 direct 产物条目数 ==="
$R/venv/bin/python - <<'PY' 2>&1 | sed 's/^/  /'
import json, os, glob
R = "/root/rivermind-data"
for p in sorted(glob.glob(f"{R}/output/*.json")):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  {os.path.basename(p):34s} 解析失败 {type(e).__name__}")
        continue
    if not isinstance(d, dict):
        print(f"  {os.path.basename(p):34s} 非 dict（{type(d).__name__}）")
        continue
    boxed = sum(1 for v in d.values() if isinstance(v, dict) and v.get("bbox"))
    print(f"  {os.path.basename(p):34s} {len(d):>5} 条｜有框 {boxed:>5}"
          f"（{boxed/max(len(d),1):.0%}）")
PY

echo; echo "=== /root 下的探针与产物 ==="
ls -la /root/*.json 2>/dev/null | head -20 | sed 's/^/  /'

echo; echo "=== GPU / 服务 ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
curl -s --max-time 4 http://127.0.0.1:8000/v1/models 2>/dev/null | head -c 120 | sed 's/^/  /'
echo
echo "=== 结束 ==="
