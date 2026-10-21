#!/bin/bash
# 无卡阶段：下载 QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ（116.64 GB）并核验完整性
# ★只要磁盘够，这一步完全不需要 GPU。
set -u
R=/root/rivermind-data
V=$R/venv/bin/python
REPO=${QWEN235_REPO:-QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ}
DIR=${QWEN235_DIR:-$R/models/Qwen3-VL-235B-A22B-Instruct-AWQ}
LOG=$R/dl235b.log

echo "=== 1) 磁盘守卫 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
avail=$(df -BG --output=avail "$R" | tail -1 | tr -dc '0-9')
echo "  可用 ${avail} GB（AWQ 需 116.64 GB，留余量取 140 GB）"
if [ "${avail:-0}" -lt 140 ]; then
  echo "  ★可用空间不足 ⇒ 中止（不触碰任何既存文件）"; exit 9
fi

echo
echo "=== 2) 是否已有下载在跑 ==="
if pgrep -f "snapshot_download|_dl235b" >/dev/null 2>&1; then
  echo "  已有下载进程，跳过启动"; exit 0
fi

echo
echo "=== 3) 分离式启动下载（不占 SSH 通道）==="
mkdir -p "$DIR"
cd "$R"
setsid nohup "$V" -u -c '
import time, sys
from modelscope import snapshot_download
t0 = time.time()
p = snapshot_download(
    "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ",
    local_dir="/root/rivermind-data/models/Qwen3-VL-235B-A22B-Instruct-AWQ")
print("DONE", p, "elapsed_s=%.0f" % (time.time() - t0), flush=True)
' > "$LOG" 2>&1 < /dev/null &
sleep 25

echo "  进程: $(pgrep -f 'snapshot_download|modelscope' | head -1 | sed 's/^/PID=/')"
echo "  日志尾 3 行:"; tail -3 "$LOG" 2>/dev/null | sed 's/^/    /'
echo "  当前目录大小: $(du -sh "$DIR" 2>/dev/null | cut -f1)"

echo
echo "=== 4) 进度查询方式（离线/随后任一时点）==="
cat <<'NOTE'
    tail -3 /root/rivermind-data/dl235b.log
    du -sh /root/rivermind-data/models/Qwen3-VL-235B-A22B-Instruct-AWQ
    pgrep -f snapshot_download >/dev/null && echo 仍在下载 || echo 已结束
NOTE
echo "=== 启动结束 ==="