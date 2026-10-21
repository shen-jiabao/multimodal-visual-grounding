#!/bin/bash
# 全量 4587 条非序数（仅在 20 条冒烟两道闸门都过后运行）
set -u
R=/root/rivermind-data
PY=$R/venv_vlx/bin/python
OUT=$R/_vlx_full.json
LOG=$R/vlx_full.log

# 同 _vlx_smoke.sh：无 nvcc ⇒ 装不上 flash_attn，回退 sdpa（非机制改动）
export VLX_ATTN="${VLX_ATTN:-sdpa}"

echo "=== 0. 前置核验 ==="
sha256sum "$R/_vlx_direct.py" | sed 's/^/  /'
echo "  期望 ABA02E7CA81916200D080921DD80EEDC77E3DB0D3D16EAEBE8FAE08B52EE6889"
echo "  权重: $(stat -c%s "$R/vlx-seek/resources/VLX-Seek-1.5-10B/model.safetensors" 2>/dev/null || echo 0) 字节（期望 20033961788）"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader 2>&1 | sed 's/^/  GPU /'

echo; echo "=== 1. 已在跑？ ==="
if pgrep -f "_vlx_direct.py" >/dev/null; then
  echo "  ● 已有实例在跑，本次不重复启动"
else
  cd "$R"
  nohup "$PY" -u _vlx_direct.py --out "$OUT" >> "$LOG" 2>&1 &
  echo "  已后台启动 PID=$!"
fi

sleep 45
echo; echo "=== 2. 45 秒后 ==="
tail -8 "$LOG" 2>/dev/null | sed 's/^/  /'
echo "  产物 $(stat -c%s "$OUT" 2>/dev/null || echo 0) 字节"
echo "=== 结束 ==="
