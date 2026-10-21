#!/bin/bash
# 清掉重复 monitor（只保留启动时报告的那个 4754），避免两者同时写同一个 .tmp
set -u
R=/root/rivermind-data

echo "=== 当前 monitor ==="
pgrep -af "_q38_monitor.sh" | sed 's/^/  /'

# 找出最早启动的那个（= 脚本自报的正式 monitor），其余杀掉
pids=$(pgrep -f "_q38_monitor.sh" | sort -n)
keep=$(echo "$pids" | head -1)
echo "  保留: $keep"
for p in $pids; do
  [ "$p" = "$keep" ] && continue
  echo "  杀掉重复: $p"
  kill "$p" 2>/dev/null
done
sleep 2
echo "  剩余 monitor: $(pgrep -cf _q38_monitor.sh || echo 0)"

echo
echo "=== worker 12 进度（缺的 3 块都在它手上）==="
pgrep -af "_q38_w.sh" | sed 's/^/  /' || echo "  （worker 已全部结束）"
P=$R/models/Qwen3.8-27B/.parts
for f in 00014 00016 00018; do
  n="$P/model-$f-of-00018.safetensors.part_3"
  sz=$(stat -c%s "$n" 2>/dev/null || echo 0)
  echo "  model-$f 块3: $(( sz / 1048576 )) MB"
done
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
