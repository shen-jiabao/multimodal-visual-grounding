#!/bin/bash
# 等 D 全量跑完并清点（分离式运行的配套等待器）
set -u
R=/root/rivermind-data
LOG=$R/d_full_run.log
for i in $(seq 1 600); do
  pgrep -f '_qwen_full_pipeline.py' >/dev/null || break
  sleep 30
done
echo "管线已退出（等待约 $((i*30)) 秒）"
echo; echo "=== 日志尾 4 行 ==="
tail -4 "$LOG"
echo; echo "=== 产物清点（对照作用范围）==="
"$R/venv/bin/python" - "$R/output/qwen32b_d3_full.json" "$R/_d_scope.json" <<'PYEOF'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1], encoding="utf-8"))
s = set(json.load(open(sys.argv[2], encoding="utf-8")))
errs = {k for k, v in d.items() if v.get("err")}
nb = sum(1 for v in d.values() if v.get("bbox"))
print(f"  产物 {len(d)}｜作用范围 {len(s)}｜缺 {len(s - set(d))}"
      f"｜清单外 {len(set(d) - s)}｜err {len(errs)}")
print(f"  有框 {nb}｜输出[]无框 {len(d) - nb - len(errs)}")
for k, n in Counter(v["err"][:60] for v in d.values() if v.get("err")).most_common(4):
    print(f"    err × {n}: {k}")
PYEOF
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== 结束 ==="