#!/bin/bash
# 建数据集目录骨架（_rc.py --sync 只建到 Images 一级，子模态目录需预先建好）。
set -u
R=/root/rivermind-data
D=$R/dataset/Images
mkdir -p "$D"/visible "$D"/infrared "$D"/depth
echo "已建："
ls -ld "$D"/* | sed 's/^/  /'
echo
echo "--- 目标盘余量 ---"
df -h "$R" | tail -1 | sed 's/^/  /'
echo "--- 连通性：写一个探针文件再删掉 ---"
if touch "$D/visible/.probe" 2>/dev/null; then
  echo "  ✓ $D/visible 可写"
  rm -f "$D/visible/.probe"
else
  echo "  ★ $D/visible 不可写"
fi
echo "=== DONE ==="
