#!/bin/bash
# bf16 下载诊断：区分「在下载 / 在合并 / 卡死」三种状态
set -u
R=/root/rivermind-data
D=$R/models/Qwen3-VL-32B-Instruct

echo "=== 当前时间 / 日志尾 ==="
date '+  now %H:%M:%S'
tail -5 "$R/q32bf16_dl.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 目录内容（含 .parts 与 .tmp） ==="
ls -la "$D" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 各分片 / 暂存体积 ==="
for f in "$D"/model-*-of-00014.safetensors; do
  [ -e "$f" ] || continue
  s=$(stat -c%s "$f" 2>/dev/null || echo 0)
  echo "  $(basename "$f")  已并 $((s/1048576)) MB"
done
for p in "$D"/*.parts; do
  [ -d "$p" ] || continue
  tot=0
  for q in "$p"/part_*; do
    [ -f "$q" ] && tot=$((tot + $(stat -c%s "$q")))
  done
  echo "  $(basename "$p")/  暂存合计 $((tot/1048576)) MB"
  ls -1 "$p" | head -3 | sed 's/^/      /'
done

echo; echo "=== curl 进程（在传还是卡住） ==="
pgrep -af "curl .*Qwen3-VL-32B-Instruct" | head -10 | cut -c1-130 | sed 's/^/  /'
echo "  curl 数 $(pgrep -cf 'curl .*Qwen3-VL-32B-Instruct' || true)"

echo; echo "=== 60 秒体积增量 ==="
t1=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 60
t2=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
echo "  $(( ${t1:-0}/1048576 )) MB → $(( ${t2:-0}/1048576 )) MB"
awk -v a="${t1:-0}" -v b="${t2:-0}" 'BEGIN{
  d=(b-a)/60;
  printf "  60 秒增量 %.2f MB（%.3f MB/s）\n", (b-a)/1048576, d/1048576;
  if ((b-a) < 1048576) print "  ★近 1 分钟增量 < 1 MB ⇒ 基本停滞"
  else print "  正常推进"}'
echo "=== 结束 ==="
