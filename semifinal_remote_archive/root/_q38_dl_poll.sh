#!/bin/bash
# 看权重下载进度
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B

if pgrep -f "_q38_dl.sh" >/dev/null 2>&1; then
  echo "状态: 主脚本运行中 pid $(pgrep -f _q38_dl.sh | tr '\n' ' ')"
else
  echo "状态: 主脚本已结束（worker 可能仍在跑）"
fi
echo "  curl 分片进程数: $(pgrep -cf 'Qwen3.8-27B/resolve' 2>/dev/null || echo 0)"

echo
echo "--- _q38_dl.out（末 20 行）---"
tail -20 "$R/_q38_dl.out" 2>/dev/null | sed 's/^/  /'

echo
echo "--- q38_dl.log（末 8 行）---"
tail -8 "$R/q38_dl.log" 2>/dev/null | sed 's/^/  /' || echo "  （尚无）"

echo
echo "--- 已就位分片 ---"
done_n=$(ls -1 "$D"/model-*-of-*.safetensors 2>/dev/null | wc -l)
psz=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
echo "  完整分片 $done_n / 18"
echo "  目录合计 $(( ${psz:-0}/1048576 )) MB"

# 40 秒实测速率
a=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 40
b=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
awk -v a="${a:-0}" -v b="${b:-0}" 'BEGIN{
  r=(b-a)/40/1048576
  printf "  实测速率 %.2f MB/s", r
  if (r>0) printf "   剩余约 %.0f 分钟\n", (53248-(b/1048576))/r/60
  else printf "\n"
}'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
