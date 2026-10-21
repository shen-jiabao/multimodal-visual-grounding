#!/bin/bash
# 等并行下载结束 → 核大小与官方 sha256
set -u
R=/root/rivermind-data
F=$R/vlx-seek/resources/VLX-Seek-1.5-10B/model.safetensors
EXP=20033961788
SHA=abbbc2d103d76bf00b032bb09a9b833ff089890a763c4874a7f28f98c62af351

echo "=== 等待并行下载结束（每 30 秒轮询，最多约 20 分钟） ==="
for i in $(seq 1 40); do
  if ! pgrep -f "_vlx_par_run.sh" >/dev/null; then
    echo "  第 $((i*30)) 秒：worker 已全部退出"
    break
  fi
  tot=$(du -sb "$R/vlx-seek/resources/VLX-Seek-1.5-10B/parts" 2>/dev/null | awk '{print $1}')
  echo "  第 $((i*30)) 秒：进行中，分片合计 $(( ${tot:-0} / 1048576 )) MB"
  sleep 30
done

echo; echo "=== 下载日志 ==="
tail -16 "$R/vlx_par.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 产物核验 ==="
if [ ! -f "$F" ]; then
  echo "  ○ 尚无 model.safetensors（可能仍在合并/校验，或已失败）"
  ls -l "$R/vlx-seek/resources/VLX-Seek-1.5-10B/" 2>/dev/null | sed 's/^/  /'
  exit 0
fi
sz=$(stat -c%s "$F")
echo "  大小 $sz / 期望 $EXP → $([ "$sz" -eq "$EXP" ] && echo 一致 || echo '★不一致')"
echo "  校验 sha256（19GB，约 1-2 分钟）..."
got=$(sha256sum "$F" | awk '{print $1}')
echo "  官方 $SHA"
echo "  实得 $got"
[ "$got" = "$SHA" ] && echo "  ★sha256 通过：权重完整且与官方一致" \
                    || echo "  ✗sha256 不符"
echo; echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
