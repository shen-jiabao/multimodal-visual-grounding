#!/bin/bash
# 最后对照：独立跑一个 pip download（清华源、--no-cache-dir、独立目录）测真实速率。
# 不干扰正在跑的构建（--no-cache-dir 避开缓存锁）。
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data
P=$R/venv_q38/bin/python
D=$R/tmp/probe

rm -rf "$D"; mkdir -p "$D"
"$P" -m pip download --no-cache-dir --no-deps --disable-pip-version-check \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  flashinfer-cubin==0.6.12 -d "$D" > "$R/tmp/probe.log" 2>&1 &
pid=$!

sleep 25
sz=$(du -sb "$D" 2>/dev/null | cut -f1)
awk -v s="${sz:-0}" 'BEGIN{printf "  独立 pip（清华源）25 秒下了 %.2f MB ⇒ %.2f MB/s\n", s/1048576, s/1048576/25}'
kill "$pid" 2>/dev/null
wait "$pid" 2>/dev/null
ls -l "$D" 2>/dev/null | sed 's/^/  /'

echo
echo "=== 主构建日志尾（含进度条）==="
tail -c 1500 "$R/q38_env.log" | tr '\r' '\n' | tail -6 | sed 's/^/  /'

echo
echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
