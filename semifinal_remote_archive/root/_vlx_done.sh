#!/bin/bash
# 等全量跑完 → 打印终态与最终计数
set -u
R=/root/rivermind-data
PY=$R/venv_vlx/bin/python
LOG=$R/vlx_full.log

echo "=== 等待 _vlx_direct.py 退出（每 20 秒轮询，最多约 20 分钟） ==="
for i in $(seq 1 60); do
  if ! pgrep -f "_vlx_direct.py" >/dev/null; then
    echo "  第 $((i*20)) 秒：进程已退出"
    break
  fi
  sleep 20
done

echo; echo "=== 日志尾 ==="
tail -6 "$LOG" | sed 's/^/  /'

echo; echo "=== 产物终态 ==="
"$PY" - <<'PYEOF'
import json
from pathlib import Path
R = Path("/root/rivermind-data")
f = R / "_vlx_full.json"
d = json.loads(f.read_text(encoding="utf-8"))
n = len(d)
boxed = [k for k, v in d.items() if v.get("bbox")]
errs = [k for k, v in d.items() if v.get("err")]
print(f"  条数 {n}  ｜有框 {len(boxed)}（{len(boxed)/n:.1%}）"
      f"｜无框 {n-len(boxed)}（{(n-len(boxed))/n:.1%}）｜err {len(errs)}")
nc = sorted(v.get("n_cand") or 0 for v in d.values())
print(f"  候选数 p10 {nc[len(nc)//10]} / p50 {nc[len(nc)//2]} / "
      f"min {nc[0]} / max {nc[-1]}")
low = sum(1 for x in nc if x < 50)
print(f"  候选数 <50 的条目 {low}（这些图上可选的余地小，是噪声源）")
print(f"  文件 {f}  {f.stat().st_size/1e6:.1f} MB")
PYEOF
echo "=== 结束 ==="
