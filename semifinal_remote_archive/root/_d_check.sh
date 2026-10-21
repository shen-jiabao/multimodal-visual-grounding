#!/bin/bash
# 查小样运行状态：进程是否幸存、产物多少、服务是否在
set -u
R=/root/rivermind-data
echo "=== 管线进程 ==="
pgrep -af 'qwen_full_pipeline' | grep -v 'bash -c' || echo "  (已不在 ⇒ 本地超时断连时被 SIGHUP 杀掉了)"
echo; echo "=== 日志尾部 ==="
tail -3 "$R/d_sample_run.log" 2>/dev/null || echo "  (无日志)"
echo; echo "=== 产物 ==="
"$R/venv/bin/python" - "$R/output/qwen32b_d3_sample.json" <<'PYEOF'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("  产物不存在"); raise SystemExit
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as ex:
    print(f"  产物非完整 JSON（就地重写中）：{type(ex).__name__}"); raise SystemExit
nb = sum(1 for v in d.values() if v.get("bbox"))
ne = sum(1 for v in d.values() if v.get("err"))
print(f"  条数 {len(d)}｜有框 {nb}｜err {ne}")
PYEOF
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo; echo "=== vLLM ==="
pgrep -f "vllm serve" >/dev/null && echo "  在服务" || echo "  ★服务不在"
echo; echo "=== 磁盘 ==="
df -h /root/rivermind-data | tail -1