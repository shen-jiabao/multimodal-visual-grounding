#!/bin/bash
# 轮询等待全量推理结束（最多约 25 分钟），随后报告进程、日志尾部与最终条目数。
# 为什么要等"进程退出"而不是只看条目数：管线结束前会再写一次 checkpoint，
# 早一步取数就是把未完成状态当结论。
set -u
PY=/root/rivermind-data/venv/bin/python
LOG=/root/rivermind-data/qwen32b_run.log

for i in $(seq 1 150); do
  if ! pgrep -f "_qwen_full_pipeline" >/dev/null; then
    echo "管线已退出（第 ${i} 次轮询，约 $((i * 10)) 秒）"
    break
  fi
  sleep 10
done

echo "=== 结束状态 ==="
pgrep -af "_qwen_full_pipeline" || echo "  无管线进程（已结束）"
echo
echo "--- 日志尾部 ---"
tail -6 "$LOG" 2>/dev/null || echo "  无日志"
echo
echo "--- 产物 ---"
ls -l /root/rivermind-data/output/qwen32b_direct.json
$PY - <<'PYEOF'
import json
p = "/root/rivermind-data/output/qwen32b_direct.json"
txt = open(p, encoding="utf-8").read()
try:
    d = json.loads(txt)
    tag = "整体可解析"
except Exception:
    d, _end = json.JSONDecoder().raw_decode(txt)
    tag = "就地重写中（已取首个完整值）"
nb = sum(1 for v in d.values() if v.get("bbox"))
ne = sum(1 for v in d.values() if v.get("err"))
print("条目 %d（目标 4587）| 有框 %d | err %d | %s"
      % (len(d), nb, ne, tag))
PYEOF