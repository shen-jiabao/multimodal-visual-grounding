#!/bin/bash
# D 全量进度（纯读，不改任何东西）
set -u
R=/root/rivermind-data
LOG=$R/d_full_run.log
echo "=== 进度 ==="
tail -2 "$LOG" 2>/dev/null || echo "  (无日志)"
echo "  已跑图数: $(grep -c '^\[[0-9]*/707\]' "$LOG" 2>/dev/null || echo 0)/707"
echo; echo "=== 产物 ==="
"$R/venv/bin/python" - "$R/output/qwen32b_d3_full.json" "$R/_d_scope.json" <<'PYEOF'
import json, os, sys
p, sp = sys.argv[1], sys.argv[2]
s = set(json.load(open(sp, encoding="utf-8")))
if not os.path.exists(p):
    print("  产物尚未落盘"); raise SystemExit
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as ex:
    print(f"  产物就地重写中（{type(ex).__name__}）"); raise SystemExit
errs = sum(1 for v in d.values() if v.get("err"))
nb = sum(1 for v in d.values() if v.get("bbox"))
print(f"  已出 {len(d)}/{len(s)}｜有框 {nb}｜输出[] {len(d)-nb-errs}｜err {errs}")
PYEOF
echo; echo "=== 进程 / GPU ==="
pgrep -f '_qwen_full_pipeline.py' | head -1 | sed 's/^/  PID /' || echo "  (已退出)"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  /'
echo; echo "=== vLLM ==="
pgrep -f 'vllm serve' >/dev/null && echo "  在服务" || echo "  ★不在"
echo "=== 结束 ==="