#!/bin/bash
# 取 Qwen3.8 环境构建的自检结果（纯 bash，避免 PowerShell 破坏内联命令）
set -u
R=/root/rivermind-data
for i in $(seq 1 30); do
  pgrep -f _q38_build.sh >/dev/null || break
  sleep 10
done
echo "=== 构建进程：$(pgrep -cf _q38_build.sh || true) 个 ==="
echo
echo "=== 第 4 步起（版本自检 / registry / CUDA / 磁盘） ==="
sed -n '/=== 4)/,$p' "$R/q38_env.log" 2>/dev/null | cut -c1-170 | sed 's/^/  /'
echo
echo "=== 关键结论摘取 ==="
grep -E "Qwen3_5|已登记|未登记|torch|transformers|vllm |causal_conv1d|flash_linear" \
  "$R/q38_env.log" 2>/dev/null | tail -14 | cut -c1-150 | sed 's/^/  /'
echo
echo "=== venv 是否建成 ==="
ls -d $R/venv_q38 2>/dev/null && du -sh $R/venv_q38 2>/dev/null | sed 's/^/  /'
df -h $R | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="