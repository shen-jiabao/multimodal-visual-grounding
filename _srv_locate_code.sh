#!/usr/bin/env bash
# 定位云端代码目录与关键产物（只读）
set +e

echo "=== candidate code dirs (code27b / infer.py / consensus_vote.py) ==="
find /root /workspace /data /hy-tmp -maxdepth 6 \
     \( -name 'code27b' -o -name 'consensus_vote.py' -o -name 'infer.py' \
        -o -name 'make_sub.py' -o -name 'predictions_27b.json' \) 2>/dev/null \
     | head -n 60

echo
echo "=== /root top level ==="
ls -la /root 2>/dev/null

echo
echo "=== /root/autodl-tmp/rivermind-data tree (depth 3) ==="
find /root/autodl-tmp/rivermind-data -maxdepth 3 2>/dev/null | head -n 200

echo
echo "=== /root/autodl-tmp/rivermind-data size ==="
du -sh /root/autodl-tmp/rivermind-data 2>/dev/null
du -sh /root/autodl-tmp/rivermind-data/* 2>/dev/null

echo
echo "=== done ==="
