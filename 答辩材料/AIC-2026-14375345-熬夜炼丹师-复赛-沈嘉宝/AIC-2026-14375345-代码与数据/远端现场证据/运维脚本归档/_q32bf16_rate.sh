#!/bin/bash
# 并发下载：速率实测（也给一次"是否该调并发数"的判断依据）
set -u
R=/root/rivermind-data
D=$R/models/Qwen3-VL-32B-Instruct
TOT=66726519322

echo "=== 进程 ==="
echo "  curl 数 $(pgrep -cf 'Qwen3-VL-32B-Instruct/resolve' || true)"
pgrep -f "_q32_monitor" >/dev/null && echo "  ● monitor 在跑" || echo "  ○ monitor 不在"
echo "  worker 数 $(pgrep -cf '_q32_w.sh' || true)"

echo; echo "=== 日志尾 ==="
tail -6 "$R/q32bf16_par.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 已就位分片 ==="
n=0
for f in "$D"/model-*-of-00014.safetensors; do
  [ -f "$f" ] || continue
  n=$(( n + 1 ))
  echo "  ✓ $(basename "$f")  $(( $(stat -c%s "$f") / 1048576 )) MB"
done
echo "  就位 $n / 14"

echo; echo "=== 残片 ==="
t=0
for f in "$D"/.parts/*.part_*; do
  [ -f "$f" ] || continue
  t=$(( t + $(stat -c%s "$f") ))
done
echo "  残片合计 $(( t / 1048576 )) MB（$(ls -1 "$D"/.parts 2>/dev/null | wc -l) 个文件）"

echo; echo "=== 80 秒速率实测 ==="
t1=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 80
t2=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
awk -v a="${t1:-0}" -v b="${t2:-0}" -v e="$TOT" 'BEGIN{
  r=(b-a)/80/1048576;
  printf "  %d MB → %d MB　速率 %.2f MB/s\n", a/1048576, b/1048576, r;
  if (r>0) printf "  剩余约 %.0f 分钟\n", (e-b)/1048576/r/60;
  printf "  进度 %.1f%%\n", 100*b/e}'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
