#!/bin/bash
# 第二批枚举（截断受影响、且第一批未覆盖的 19 对）：过闸门后 setsid nohup 起。
# 与第一批的唯一区别是环境变量：QWEN_ENUM_TARGETS / QWEN_ENUM_OUT 指向第二批文件。
set -u
PY=/root/miniconda3/envs/myconda/bin/python
LOG=/srv/output/enum_retry2.log
MODEL=/srv/models/Qwen3-VL-8B-Instruct
WORKER=/root/_enum_retry.py
fail=0

echo "===== 闸门 1) 模型权重 ====="
n_sf=$(ls "$MODEL"/*.safetensors 2>/dev/null | wc -l)
echo "safetensors=$n_sf"
if [ "$n_sf" -lt 4 ]; then echo "FAIL: 模型权重不完整"; fail=1; fi

echo "===== 闸门 2) 第二批目标集 / 键集 / 引用图 ====="
$PY - <<'PYEOF'
import json
import sys
from pathlib import Path
T = Path("/srv/output/_enum_targets2.json")
D = Path("/srv/dataset")
if not T.exists():
    print("FAIL: 缺少 _enum_targets2.json")
    sys.exit(1)
tg = json.loads(T.read_text(encoding="utf-8"))
q = json.loads((D / "queries" / "queries.json").read_text(encoding="utf-8"))
keys = list(tg["target"]) + list(tg["control"])
miss = [k for k in keys if k not in q]
bad = [p for p in {q[k]["visible"] for k in keys if k in q} if not (D / p).exists()]
print("target", len(tg["target"]), "| control", len(tg["control"]),
      "| 键缺失", len(miss), "| 图缺失", len(bad))
sys.exit(1 if (miss or bad) else 0)
PYEOF
if [ $? -ne 0 ]; then fail=1; fi

echo "===== 闸门 3) 显存 ====="
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "空闲显存 ${free_mib} MiB"
if [ "$free_mib" -lt 20000 ]; then echo "FAIL: 显存不足"; fail=1; fi

echo "===== 结论 ====="
if [ $fail -ne 0 ]; then echo "BLOCKED: 拒绝启动"; exit 2; fi
echo "PASS: 可以启动"

if pgrep -f "_enum_retry.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _enum_retry.py | head -1)"
  exit 0
fi

cd /root
QWEN_ENUM_TARGETS=/srv/output/_enum_targets2.json \
QWEN_ENUM_OUT=/srv/output/_enum_raw2.json \
setsid nohup $PY -u "$WORKER" > "$LOG" 2>&1 < /dev/null &
sleep 25
echo "已启动 PID=$(pgrep -f _enum_retry.py | head -1)"
echo "--- 日志首 20 行 ---"
head -n 20 "$LOG"
