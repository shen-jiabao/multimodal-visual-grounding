#!/bin/bash
# 复赛 LocateAnything-3B 全量启动器：带闸门，任一前置不满足即拒绝启动。
# 设计原则与 8B 那套一致：数据/环境不齐时不跑，避免把失败写进产物。
#
# 解释器用 /srv/venv-la2（挂 myconda 的 transformers 4.57.6 + 本地补 decord/lmdb/peft）。
# 不用 /srv/venv-locate：其 transformers 4.57.1 的 AutoProcessor 实测无法导入。
set -u
VPY=/srv/venv-la2/bin/python
MODEL=/srv/models/LocateAnything-3B
DATA=/srv/dataset
OUT=/srv/output/locate_bbox_all.json
SCRIPT=/root/_locate_full.py
LOG=/srv/output/locate_full_stdout.log
fail=0

echo "===== 闸门 1) 模型权重 ====="
n_sf=$(ls $MODEL/*.safetensors 2>/dev/null | wc -l)
n_inc=$(ls $MODEL/*.incomplete 2>/dev/null | wc -l)
echo "safetensors=$n_sf (应 2)  incomplete=$n_inc (应 0)"
[ "$n_sf" -ne 2 ] && fail=1
[ "$n_inc" -ne 0 ] && fail=1
[ -f $MODEL/model.safetensors.index.json ] || { echo "缺 index.json"; fail=1; }

echo
echo "===== 闸门 2) 解释器依赖 ====="
$VPY - <<'PY'
import sys
need = ("torch", "transformers", "peft", "decord", "lmdb", "PIL")
bad = []
for m in need:
    try:
        mod = __import__(m)
        print(f"  {m:13s}", getattr(mod, "__version__", "?"))
    except Exception as e:
        print(f"  {m:13s} FAIL {type(e).__name__}")
        bad.append(m)
from transformers import AutoProcessor  # noqa: F401
print("  AutoProcessor 可导入")
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && fail=1

echo
echo "===== 闸门 3) 数据：queries + 引用图完整可读 ====="
$VPY - <<'PY'
import json, sys
from pathlib import Path
from PIL import Image
D = Path("/srv/dataset")
q = json.loads((D/"queries"/"queries.json").read_text(encoding="utf-8"))
names = sorted({Path(v["visible"]).name for v in q.values()})
vd = D/"Images"/"visible"
miss = [n for n in names if not (vd/n).exists()]
bad = []
for n in names:
    if n in miss: continue
    try:
        Image.open(vd/n).load()
    except Exception:
        bad.append(n)
print(f"  queries {len(q)} | 引用图 {len(names)} | 缺失 {len(miss)} | 不可读 {len(bad)}")
if miss: print("  缺失样例", miss[:5])
if bad:  print("  不可读样例", bad[:5])
sys.exit(1 if (miss or bad) else 0)
PY
[ $? -ne 0 ] && fail=1

echo
echo "===== 闸门 4) GPU 必须空闲（须等 8B 主任务跑完）====="
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$')
if [ -n "$PIDS" ]; then
  echo "  GPU 被占用: $PIDS"
  for p in $PIDS; do ps -p $p -o pid,etime,cmd --no-headers 2>/dev/null | cut -c1-120; done
  fail=1
else
  echo "  GPU 空闲"
fi

echo
if [ $fail -ne 0 ]; then echo "BLOCKED: 前置未满足，拒绝启动"; exit 1; fi
echo "PASS: 可以启动 LocateAnything 全量"

if pgrep -f "_locate_full.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _locate_full.py | head -1)，不重复启动"; exit 0
fi
cd /root
setsid nohup nice -n 5 $VPY -u $SCRIPT > $LOG 2>&1 < /dev/null &
sleep 8
pid=$(pgrep -f _locate_full.py | head -1)
echo "已启动 PID=$pid"
echo "--- 日志首 12 行 ---"
head -n 12 $LOG
