#!/bin/bash
# 下载速率诊断 + worker 结果格式 + flash_attn 结果
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
RES=$VLX/resources
F=$RES/VLX-Seek-1.5-10B/model.safetensors

echo "=== A. 下载速率（20 秒采样） ==="
if [ -f "$F" ]; then
  s1=$(stat -c%s "$F"); sleep 20; s2=$(stat -c%s "$F")
  echo "  20 秒内 $(( (s2-s1)/1024 )) KB  → 约 $(( (s2-s1)/20/1024 )) KB/s"
  echo "  当前 $(( s2/1024/1024 )) MB / 约 20480 MB（$(python3 -c "print(f'{$s2/20480/1048576*100:.1f}%')")）"
else
  echo "  文件尚未创建"
fi
pgrep -af "curl.*VLX-Seek" | head -3 | sed 's/^/  /'

echo; echo "=== B. 日志尾 ==="
tail -5 "$R/vlx_dl_model.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== C. worker: _build_result_bbox_list（★结果字段格式） ==="
sed -n '105,150p' "$VLX/vlx_seek_worker.py" | sed 's/^/  /'

echo; echo "=== D. worker: predict 的返回结构 ==="
sed -n '240,265p' "$VLX/vlx_seek_worker.py" | sed 's/^/  /'

echo; echo "=== E. flash_attn 重试结果 ==="
pgrep -f "flash_attn" >/dev/null && echo "  ● 仍在编译" || echo "  ○ 已结束"
tail -6 "$R/vlx_flash.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== F. 是否有 ModelScope 镜像（备选下载源） ==="
curl -sS --max-time 25 "https://modelscope.cn/api/v1/models/omlab/VLX-Seek-1.5-10B" 2>/dev/null \
  | head -c 300 | sed 's/^/  /'
echo
echo "=== 结束 ==="
