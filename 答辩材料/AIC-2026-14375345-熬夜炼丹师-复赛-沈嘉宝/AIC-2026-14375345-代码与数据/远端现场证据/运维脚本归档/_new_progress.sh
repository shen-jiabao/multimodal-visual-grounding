#!/bin/bash
# 全量进度查询。
PY=/root/rivermind-data/venv/bin/python
OUT=/root/rivermind-data/output/qwen32b_direct.json
LOG=/root/rivermind-data/qwen32b_run.log

echo "=== 运行进程 ==="
pgrep -af "_qwen_full_pipeline" | head -2 || echo "  未在运行"
echo
echo "=== 运行日志尾部 ==="
tail -14 "$LOG" 2>/dev/null || echo "  无日志"
echo
echo "=== 已完成条目数 ==="
$PY - <<'PYEOF'
import json, os
p = "/root/rivermind-data/output/qwen32b_direct.json"
if not os.path.exists(p):
    print("  输出文件尚未生成")
else:
    d = json.load(open(p, encoding="utf-8"))
    nb = sum(1 for v in d.values() if v.get("bbox"))
    ne = sum(1 for v in d.values() if v.get("err"))
    print("  已写 " + str(len(d)) + " 条 | 有框 " + str(nb) + " | err " + str(ne)
          + " | 目标 4587")
    if len(d):
        print("  完成度 " + format(len(d)/4587, ".1%"))
PYEOF
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
curl -s --max-time 5 http://127.0.0.1:8000/metrics 2>/dev/null | grep -E "^vllm:(num_requests_running|num_requests_waiting|request_success_total)" | head -5