#!/bin/bash
# VLX-Seek 进度 + 下载 WeDetect 检测器权重
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
RES=$VLX/resources
mkdir -p "$RES"

echo "=== D. WeDetect 检测器（base_uni，后台） ==="
if [ -f "$RES/wedetect_base_uni.pth" ]; then
  echo "  已有 $(du -h "$RES/wedetect_base_uni.pth" | cut -f1)"
else
  nohup curl -sSL --retry 5 -C - -o "$RES/wedetect_base_uni.pth" \
    "https://hf-mirror.com/fushh7/WeDetect/resolve/main/wedetect_base_uni.pth" \
    > "$R/vlx_dl_det.log" 2>&1 &
  echo "  已后台启动 PID=$!"
fi

echo; echo "=== A. pip 进度 ==="
pgrep -f "_vlx_pip" >/dev/null && echo "  ● 仍在装" || echo "  ○ 已结束"
tail -6 "$R/vlx_pip.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== B. 权重下载进度（目标约 20G） ==="
pgrep -f "_vlx_dl_model" >/dev/null && echo "  ● 仍在下" || echo "  ○ 已结束"
D=$RES/VLX-Seek-1.5-10B
[ -d "$D" ] && du -sh "$D" 2>/dev/null | sed 's/^/  已落 /'
ls -1 "$D" 2>/dev/null | sed 's/^/    /'
tail -3 "$R/vlx_dl_model.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
