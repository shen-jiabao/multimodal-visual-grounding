#!/bin/bash
cd /root/autodl-tmp/work/code27b || { echo NO_DIR; exit 0; }
echo "=== code27b 全部文件（排除缓存） ==="
find . -type f ! -path "*/__pycache__/*" ! -name "*.pyc" | sort
echo "=== outputs ==="
ls -la outputs/
echo "=== outputs/old_0103 ==="
ls -la outputs/old_0103/ 2>/dev/null
echo "=== work 顶层 ==="
ls -la /root/autodl-tmp/work/
echo "=== 各文件大小合计 ==="
du -sh /root/autodl-tmp/work/code27b
