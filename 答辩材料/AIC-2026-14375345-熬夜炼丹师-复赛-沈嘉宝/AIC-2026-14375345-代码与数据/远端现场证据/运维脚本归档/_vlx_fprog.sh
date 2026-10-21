#!/bin/bash
# VLX 全量实时进度：产物计数 + 速率 + ETA + GPU
set -u
R=/root/rivermind-data
PY=$R/venv_vlx/bin/python
F=$R/_vlx_full.json
LOG=$R/vlx_full.log

echo "=== 1. 进程 ==="
RUN=0
if pgrep -f "_vlx_direct.py" >/dev/null; then
  RUN=1
  echo "  ● 在跑  PID $(pgrep -f '_vlx_direct.py' | tr '\n' ' ')"
else
  echo "  ○ 已结束（或未运行）"
fi

echo; echo "=== 2. 日志尾 ==="
grep -E "^  \[[0-9]+/4587\]|模型已加载|待跑|完成：" "$LOG" 2>/dev/null | tail -4 | sed 's/^/  /'

echo; echo "=== 3. 进度与速率 ==="
# 日志口径（每 10 条打一行，最准）；产物每 25 条才落盘，两分钟采样才可靠
last=$(grep -E "^  \[[0-9]+/4587\]" "$LOG" 2>/dev/null | tail -1)
echo "  日志末行：${last:-（无）}"
Ln=$(echo "$last" | sed -n 's/.*\[\([0-9]*\)\/4587\].*/\1/p')
Lr=$(echo "$last" | sed -n 's/.*\] \([0-9.]*\) 条\/s.*/\1/p')
if [ -n "${Ln:-}" ]; then
  echo "  日志进度  $Ln / 4587  ( $(awk -v a="$Ln" 'BEGIN{printf "%.1f", 100*a/4587}') % )"
  awk -v n="$Ln" -v r="${Lr:-0}" 'BEGIN{
    if (r>0) printf "  按日志速率 %.2f 条/s ⇒ 剩余约 %.0f 分钟\n", r, (4587-n)/r/60}'
fi

n1=0
[ -f "$F" ] && n1=$("$PY" -c "
import json;print(len(json.load(open('$F'))))" 2>/dev/null || echo 0)
if [ "$RUN" -eq 1 ]; then
  sleep 120
else
  echo "  （进程已退出，跳过两点采样）"
fi
[ -f "$F" ] && n2=$("$PY" -c "
import json;print(len(json.load(open('$F'))))" 2>/dev/null || echo 0)
echo "  产物条数  $n1 → $n2 / 4587"
awk -v a="$n1" -v b="$n2" 'BEGIN{
  d=(b-a)/120;
  if (d>0) printf "  产物口径速率 %.2f 条/s（与日志口径互为印证）\n", d;
  else print "  产物口径：本窗口未跨过 25 条落盘点（或已跑完）"}'

echo; echo "=== 4. 健康度（中途口径，含未跑到的分开显示） ==="
[ -f "$F" ] && "$PY" - <<'PYEOF'
import json
from pathlib import Path
R = Path("/root/rivermind-data")
d = json.loads((R / "_vlx_full.json").read_text(encoding="utf-8"))
n = len(d)
boxed = [k for k, v in d.items() if v.get("bbox")]
errs = [k for k, v in d.items() if v.get("err")]
print(f"  已落盘 {n} 条｜有框 {len(boxed)}（{len(boxed)/max(n,1):.1%}）"
      f"｜无框 {n-len(boxed)}（{(n-len(boxed))/max(n,1):.1%}）｜err {len(errs)}")
if errs:
    print(f"  err 样例：{[(k, d[k]['err'][:60]) for k in errs[:3]]}")
nc = [v.get("n_cand") or 0 for v in d.values()]
print(f"  候选数 min {min(nc)} / max {max(nc)}（异常低意味着检测器某图没出框）")
oi = sum(1 for v in d.values() if v.get("object_index") in (None, "<obj0>"))
print(f"  object_index 为空或 obj0 的 {oi} 条")
PYEOF

echo; echo "=== 5. GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
           --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="
