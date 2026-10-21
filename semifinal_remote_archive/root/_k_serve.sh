#!/bin/bash
# K 件（Qwen2.5-VL-72B-Instruct-AWQ）起服启动器
#   与 I 件的唯一差别 = 下面 5 个 Q38_* 环境变量（起服机制本身不复制第二份）
#   用法：bash _k_serve.sh        （后台发起，立即返回；就绪用 _k_poll.sh 查）
set -u
R=/root/rivermind-data
DL=$R/models/Qwen2.5-VL-72B-Instruct-AWQ

# ---- 前置：目录名断言（与起服脚本内的 MUST/FORBID 双重把关）----
if [ ! -d "$DL" ]; then echo "★FAIL: 权重目录不存在 $DL"; exit 9; fi
echo "权重目录 $DL"
du -sh "$DL" 2>/dev/null | sed 's/^/  /'
ls -1 "$DL"/model-*.safetensors 2>/dev/null | wc -l | sed 's/^/  分片数 /'

# ---- 停掉旧服务（本机当前无保留服务）----
pkill -f "_q38_serve_run.sh" 2>/dev/null
pkill -f "vllm serve" 2>/dev/null
sleep 4
rm -f "$R/_serve_k.log"

# ---- 后台发起（detached；就绪与否由 _k_poll.sh 以 HTTP 判定）----
setsid nohup env \
  Q38_DIR="$DL" \
  Q38_NAME=qwen25vl-72b-awq \
  Q38_MUST=Qwen2.5-VL-72B \
  Q38_FORBID=Qwen3 \
  Q38_LOG="$R/vllm_k.log" \
  Q38_PORT=8000 \
  bash "$R/_get_serve_q38.sh" > "$R/_serve_k.log" 2>&1 < /dev/null &
sleep 5
echo "已发起 PID=$(pgrep -f _get_serve_q38.sh | head -1)｜日志 $R/_serve_k.log"
echo "=== 启动段首 25 行 ==="
head -n 25 "$R/_serve_k.log" 2>/dev/null | sed 's/^/  /'
echo "=== END ==="
