#!/bin/bash
W=/root/autodl-tmp/work
echo "=== repro27b ==="
ls -la $W/repro27b/
du -sh $W/repro27b
echo "=== label1024 ==="
ls -la $W/label1024/
du -sh $W/label1024
echo "=== label1024 抽样内容 ==="
find $W/label1024 -maxdepth 2 -type f | head -8
echo "=== 是否有真值字段 ==="
find $W/label1024 -name "*.json" | head -2 | while read f; do echo "-- $f"; head -c 400 "$f"; echo; done
echo "=== code27b 总大小与文件数 ==="
du -sh $W/code27b
find $W/code27b -type f | wc -l
