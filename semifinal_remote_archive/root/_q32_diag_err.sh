#!/bin/bash
# 取第二次起服的根因（不猜，直读日志里的真实报错）
R=/root/rivermind-data
for f in "$R/serve32bf16b.log" "$R/vllm32bf16.log"; do
  echo "########## $f ##########"
  [ -f "$f" ] || { echo "  （不存在）"; continue; }
  echo "--- 行数 $(wc -l < "$f") ---"
  echo "--- 真实报错（Error / Traceback 主干 / OOM / assert） ---"
  grep -nE "Error|error|OOM|out of memory|assert|Assertion|ValueError|RuntimeError|TypeError|NotImplemented|CUDA|Failed|failed" "$f" \
    | grep -vE "^\s*$" | tail -30 | cut -c1-200 | sed 's/^/  /'
  echo "--- 加载进度最后状态 ---"
  grep -nE "Loading safetensors|Checkpoint|weights took|GiB|memory" "$f" | tail -12 | cut -c1-190 | sed 's/^/  /'
  echo
done

echo "########## 现场 GPU ##########"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
echo "########## 当前进程 ##########"
pgrep -af "vllm serve" | head -3 | cut -c1-140 | sed 's/^/  /'
echo "=== 结束 ==="