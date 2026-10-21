#!/bin/bash
# 枚举重跑启动器：先过 4 道闸门，通过后才 setsid nohup 起，避免"数据不齐硬跑"。
set -u
PY=/root/miniconda3/envs/myconda/bin/python
LOG=/srv/output/enum_retry.log
MODEL=/srv/models/Qwen3-VL-8B-Instruct
WORKER=/root/_enum_retry.py
fail=0

echo "===== 闸门 1) 模型权重 ====="
n_sf=$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)
n_inc=$(ls "$MODEL"/*.incomplete 2>/dev/null | wc -l)
echo "safetensors=$n_sf incomplete=$n_inc"
if [ "$n_sf" -lt 4 ] || [ "$n_inc" -ne 0 ]; then echo "FAIL: 模型权重不完整"; fail=1; fi

echo "===== 闸门 2) 目标集 / 键集 / 引用图 ====="
$PY - <<'PYEOF'
import json
import sys
from pathlib import Path
T = Path("/srv/output/_enum_targets.json")
D = Path("/srv/dataset")
if not T.exists():
    print("FAIL: 缺少 _enum_targets.json")
    sys.exit(1)
tg = json.loads(T.read_text(encoding="utf-8"))
q = json.loads((D / "queries" / "queries.json").read_text(encoding="utf-8"))
print("target", len(tg["target"]), "| control", len(tg["control"]),
      "| queries", len(q))
keys = list(tg["target"]) + list(tg["control"])
miss = [k for k in keys if k not in q]
imgs = {q[k]["visible"] for k in keys if k in q}
bad = [p for p in imgs if not (D / p).exists()]
print(f"引用图 {len(imgs)} | 键缺失 {len(miss)} | 图缺失 {len(bad)}")
if miss or bad:
    print("缺失样例", bad[:3])
    sys.exit(1)
sys.exit(0)
PYEOF
[ $? -ne 0 ] && fail=1

echo "===== 闸门 3) 显存 ====="
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "空闲显存 ${free_mib} MiB"
if [ "$free_mib" -lt 20000 ]; then echo "FAIL: 显存不足（需 >=20 GB）"; fail=1; fi

echo "===== 闸门 4) 采集脚本常量 ====="
grep -nE '^(MODEL_PATH|DATA|OUT|TARGETS|SAVE_EVERY|MAX_NEW_)' "$WORKER"

echo "===== 结论 ====="
if [ $fail -ne 0 ]; then echo "BLOCKED: 拒绝启动"; exit 2; fi
echo "PASS: 可以启动"

if pgrep -f "_enum_retry.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _enum_retry.py | head -1)"
  exit 0
fi

cd /root
setsid nohup $PY -u "$WORKER" > "$LOG" 2>&1 < /dev/null &
sleep 25
pid=$(pgrep -f _enum_retry.py | head -1)
echo "已启动 PID=$pid"
echo "--- 日志首 25 行 ---"
head -n 25 "$LOG"
