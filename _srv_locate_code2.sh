#!/usr/bin/env bash
# 更广范围定位提交代码与产物（只读）
set +e

echo "=== root dirs ==="
ls -la /

echo
echo "=== data mounts ==="
df -h 2>/dev/null | grep -vE 'tmpfs|overlay'

echo
echo "=== 找 code27b / consensus_vote / make_sub / predictions_* ==="
find / -maxdepth 7 \
     \( -path '/proc' -o -path '/sys' -o -path '/root/autodl-tmp/rivermind-data/uvcache' \
        -o -path '/root/autodl-tmp/rivermind-data/venv_q38' \
        -o -path '/root/autodl-tmp/rivermind-data/models' \
        -o -path '/root/miniconda3' \) -prune -o \
     \( -name 'code27b' -o -name 'consensus_vote.py' -o -name 'make_sub.py' \
        -o -name 'predictions_*.json' -o -name 'consensus_pool.json.gz' \) -print 2>/dev/null \
     | head -n 40

echo
echo "=== 找压缩包 / git 仓库 ==="
find /root /autodl-tmp -maxdepth 5 \
     \( -path '/root/autodl-tmp/rivermind-data/uvcache' \
        -o -path '/root/autodl-tmp/rivermind-data/venv_q38' \) -prune -o \
     \( -name '*.tar.gz' -o -name '*.zip' -o -name '.git' \) -print 2>/dev/null | head -n 40

echo
echo "=== output 目录 ==="
ls -la /root/autodl-tmp/rivermind-data/output 2>/dev/null

echo
echo "=== 关键文件 md5 ==="
md5sum /root/autodl-tmp/rivermind-data/dataset/queries/queries.json 2>/dev/null
md5sum /root/autodl-tmp/rivermind-data/output/*.json 2>/dev/null
ls -la /root/autodl-tmp/rivermind-data/dataset/queries 2>/dev/null

echo
echo "=== git ==="
cd /root/autodl-tmp/rivermind-data 2>/dev/null && git rev-parse --show-toplevel 2>&1 | head -n 2
cd / 2>/dev/null

echo
echo "=== done ==="
