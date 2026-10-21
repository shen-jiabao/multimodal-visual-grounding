#!/bin/bash
# 活性检查：采样 .incomplete 大小两次 + 匹配所有可能的下载进程
INC=/srv/models/LocateAnything-3B/model-00001-of-00002.safetensors.incomplete
echo "== 采样1 =="
S1=$(stat -c %s "$INC" 2>/dev/null || echo MISSING)
echo "size1=$S1"
pgrep -af -f "modelscope|snapshot_download" | grep -v pgrep | head -5
echo "== 等 30s =="
sleep 30
echo "== 采样2 =="
S2=$(stat -c %s "$INC" 2>/dev/null || echo MISSING)
echo "size2=$S2"
echo "== 活跃python进程 =="
ps aux | grep -E "python.*(modelscope|snapshot|download)" | grep -v grep | head -5
echo "== 结论 =="
if [ "$S1" != "MISSING" ] && [ "$S2" != "MISSING" ] && [ "$S2" -gt "$S1" ]; then
  echo "ACTIVE: 下载有进展 (增长 $((S2-S1)) bytes/30s)"
elif [ "$S1" = "$S2" ] && [ "$S1" != "MISSING" ]; then
  echo "STALLED: 30s 无增长"
else
  echo "UNKNOWN/END: 文件可能已改名(下载完成)或缺失"
  ls -la /srv/models/LocateAnything-3B/model-00001-of-00002.safetensors 2>/dev/null
fi