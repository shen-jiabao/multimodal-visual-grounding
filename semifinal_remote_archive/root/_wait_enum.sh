#!/bin/bash
# 等枚举重跑结束，然后打印结果摘要。用于一次性阻塞监视，无需反复轮询。
# 用法（远端）：bash /root/_wait_enum.sh [最多等待秒数]
set -u
MAX="${1:-9000}"
PY=/root/miniconda3/envs/myconda/bin/python
RAW=/srv/output/_enum_raw.json
LOG=/srv/output/enum_retry.log

t0=$(date +%s)
while pgrep -f "_enum_retry.py" > /dev/null; do
  now=$(date +%s)
  el=$((now - t0))
  if [ "$el" -ge "$MAX" ]; then
    echo "等待超时（${el}s），进程仍在运行"
    break
  fi
  sleep 60
done

echo "=== 进程状态 ==="
pgrep -af _enum_retry.py || echo "已退出"
echo "=== 日志末 12 行 ==="
tail -n 12 "$LOG"
echo "=== 产出统计 ==="
$PY - <<'PYEOF'
import json
from pathlib import Path
p = Path("/srv/output/_enum_raw.json")
raw = json.loads(p.read_text(encoding="utf-8"))
print("对数", len(raw), "| 文件", p.stat().st_size, "B")
ks = ("p1", "p2", "p3", "t1", "t2", "t3", "t4")
err = [k for k, v in raw.items() if "err" in v]
print("整对失败（图打不开）", len(err), err[:3])
n_all = sum(1 for v in raw.values()
            if max(len(v.get(k) or []) for k in ks) >= 2)
print("至少一种提示给出 >=2 框的对", n_all)
from collections import Counter
c = Counter()
for v in raw.values():
    c[min(max(len(v.get(k) or []) for k in ks), 6)] += 1
print("单提示最大框数分布(>=6 归 6):", dict(sorted(c.items())))
PYEOF
echo "=== 显存 ==="
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
