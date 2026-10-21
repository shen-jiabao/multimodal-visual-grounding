#!/bin/bash
# 删除已被 q38 超越的 32B 权重，为异质模型腾盘（★用户明确授权）
# 不删 Qwen3.8-27B —— 方案 A 要在序数层跑它
set -u
R=/root/rivermind-data
D=$R/models/Qwen3-VL-32B-Instruct

echo "=== 删前 ==="
df -h $R | tail -1 | sed 's/^/  /'
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'

if [ -d "$D" ]; then
  echo "  正在删除 $D ..."
  rm -rf "$D"
  echo "  退出码 $?"
else
  echo "  （$D 不存在，跳过）"
fi

echo; echo "=== 删后 ==="
df -h $R | tail -1 | sed 's/^/  /'
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'
echo "  可用空间：$(df -BG --output=avail $R | tail -1 | tr -dc '0-9') GB"

echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
echo "=== 结束 ==="
