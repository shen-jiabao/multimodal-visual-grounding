#!/bin/bash
# 诊断：vLLM 安装为何静默中断（venv 6.2 GB 但无 vllm 可执行、两个日志都近乎空）。
set -u
VENV=/root/rivermind-data/venv
echo "=== 1) venv 里已装了什么（大头）==="
"$VENV/bin/pip" list 2>/dev/null | grep -iE "^(torch|vllm|transformers|numpy|openai|pillow|xformers|flashinfer|ray|nvidia)" | head -30
echo
echo "=== 2) site-packages 体量前 12 名 ==="
du -sh "$VENV"/lib/python3.10/site-packages/* 2>/dev/null | sort -rh | head -12
echo
echo "=== 3) 是否有 torch 与 vllm 的残留 ==="
ls -d "$VENV"/lib/python3.10/site-packages/{torch,vllm}* 2>/dev/null || echo "  无 torch/vllm 目录"
echo
echo "=== 4) 环境限制：OOM 记录与磁盘 ==="
dmesg 2>/dev/null | tail -5 | grep -i -E "oom|kill" || echo "  dmesg 无 oom 记录（或无权限）"
df -h /root/rivermind-data | tail -1
echo
echo "=== 5) pip 缓存里已下了哪些包（说明下载到哪一步）==="
find /root/rivermind-data/pip-cache -name "*.whl" 2>/dev/null | wc -l
find /root/rivermind-data/pip-cache -name "*.whl" 2>/dev/null | sed 's#.*/##' | head -20