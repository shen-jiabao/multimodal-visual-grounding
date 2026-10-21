#!/bin/bash
# 329 条单条回补的进度
set -u
R=/root/rivermind-data
F=$R/output/m_retry.json

if pgrep -f "_qwen_full_pipeline.py" >/dev/null 2>&1; then
  echo "=== 进程 运行中 pid $(pgrep -f _qwen_full_pipeline.py | tr '\n' ' ') ==="
else
  echo "=== 进程 ★已结束 ==="
fi
echo "--- 日志末 3 行 ---"
tail -3 "$R/m_retry.log" 2>/dev/null | cut -c1-150 | sed 's/^/  /'
$R/venv_q38/bin/python - "$F" <<'PY'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("  产物尚未生成")
    raise SystemExit
d = json.load(open(p, encoding="utf-8"))
ok = sum(1 for v in d.values() if v.get("bbox"))
print(f"  回补已出 {len(d)}/329 条｜其中有框 {ok}"
      f"（{ok/max(1,len(d))*100:.1f}%）")
PY
echo "=== DONE ==="
