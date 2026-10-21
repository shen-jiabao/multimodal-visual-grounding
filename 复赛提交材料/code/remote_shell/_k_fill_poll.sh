#!/bin/bash
# 进度查询：bash _k_fill_poll.sh [产物路径] [日志路径]
set -u
R=/root/rivermind-data
O=${1:-$R/output/k72b_direct_429.json}
L=${2:-$R/k72b_fill.log}
echo "===== 产物/日志 ====="
echo "  $O"
echo "  $L"
echo "===== 进程 ====="
pgrep -af "_qwen_full_pipeline.py" | cut -c1-140 || echo "  (已结束)"
echo "===== 进度 ====="
tail -n 6 "$L" 2>/dev/null | cut -c1-150 | sed 's/^/  /'
echo "===== 产物 ====="
if [ -f "$O" ]; then
  ls -la "$O" | sed 's/^/  /'
  $R/venv/bin/python - "$O" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = [k for k, v in d.items() if v.get("bbox")]
print(f"  条目 {len(d)}｜有框 {len(ok)}｜无框 {len(d) - len(ok)}")
err = [k for k, v in d.items() if v.get("err")]
if err:
    print(f"  err {len(err)} 条，样例：{err[:3]}")
bad = [k for k, v in d.items() if v.get("bbox")
       and not all(0 <= x <= 1 for x in v["bbox"])]
print(f"  坐标越界 {len(bad)} 条")
PY
else
  echo "  (尚无产物)"
fi
echo "=== END ==="
