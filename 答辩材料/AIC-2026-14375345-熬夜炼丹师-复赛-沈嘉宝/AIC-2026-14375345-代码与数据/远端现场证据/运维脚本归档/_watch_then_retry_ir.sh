#!/bin/bash
# 等 LA 全量结束后，自动对「8B 无框的 45 条」用**红外图**补跑一轮。
# 与前面的衔接链同构：先闸门，再启动；职责分离（何时去试 vs 是否放行）。
# 日志 /srv/output/watch_retry_ir.log
LOG=/srv/output/watch_retry_ir.log
VPY=/srv/venv-la2/bin/python
PY=/root/miniconda3/envs/myconda/bin/python
exec > >(tee -a $LOG) 2>&1
echo "=== [$(date)] 监视 LA 全量，结束即跑红外补轮 ==="

n=0
while pgrep -f '_locate_full\.py' > /dev/null; do
  n=$((n + 1))
  if [ $((n % 10)) -eq 1 ]; then
    printf '[%s] 心跳 第 %d 次检查：LA 仍在运行\n' "$(date +%H:%M:%S)" "$n"
  fi
  sleep 60
done
echo "[$(date)] LA 进程已退出；等 45s 让显存释放..."
sleep 45

echo
echo "===== 闸门 1) LA 全量产物完整性 ====="
$PY - <<'PY'
import json, os
p = "/srv/output/locate_bbox_all.json"
if not os.path.exists(p):
    print("  缺失:", p); raise SystemExit(1)
d = json.load(open(p, encoding="utf-8"))
nb = sum(1 for v in d.values() if v.get("bbox"))
na = sum(1 for v in d.values() if v.get("bbox_all"))
err = sum(1 for v in d.values() if v.get("err"))
print(f"  {len(d)} 条 | 有框 {nb} | 带 bbox_all {na} | err {err}")
print("  期望 5690 条；带 bbox_all 说明跑的是已修复版脚本")
raise SystemExit(0 if len(d) == 5690 else 1)
PY
[ $? -ne 0 ] && { echo "BLOCKED: LA 产物不完整"; exit 1; }

echo
echo "===== 闸门 2) 红外图 与 无框清单 ====="
$PY - <<'PY'
import json, os, sys
from PIL import Image
q = json.load(open("/srv/dataset/queries/queries.json", encoding="utf-8"))
qids = json.load(open("/srv/output/nobox_qids.json", encoding="utf-8"))
ir = sorted({q[k]["infrared"] for k in qids})
miss = [p for p in ir if not os.path.exists(os.path.join("/srv/dataset", p))]
bad = []
for p in ir:
    if p in miss: continue
    try:
        Image.open(os.path.join("/srv/dataset", p)).load()
    except Exception:
        bad.append(p)
print(f"  待补 {len(qids)} 条 | 需要红外图 {len(ir)} 张 | 缺失 {len(miss)} | 不可读 {len(bad)}")
if miss: print("  缺失样例", miss[:5])
if bad:  print("  不可读样例", bad[:5])
sys.exit(1 if (miss or bad) else 0)
PY
[ $? -ne 0 ] && { echo "BLOCKED: 红外图不齐"; exit 1; }

echo
echo "===== 闸门 3) GPU 空闲 ====="
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$')
if [ -n "$PIDS" ]; then
  echo "  GPU 仍被占用: $PIDS"; echo "BLOCKED: 拒绝启动"; exit 1
fi
echo "  GPU 空闲"

echo
echo "===== 闸门 4) 解释器依赖 ====="
$VPY - <<'PY'
import transformers, peft, decord, lmdb
from transformers import AutoProcessor  # noqa: F401
print("  OK:", transformers.__version__, peft.__version__, decord.__version__, lmdb.__version__)
PY
[ $? -ne 0 ] && { echo "BLOCKED: 依赖异常"; exit 1; }

echo
echo "[$(date)] PASS，启动红外补轮..."
cd /root
setsid nohup $VPY -u /root/_retry_ir.py > /srv/output/retry_ir_stdout.log 2>&1 < /dev/null &
sleep 12
echo "  PID: $(pgrep -f '_retry_ir\.py' | tr '\n' ' ')"
echo "--- 日志前 8 行 ---"
head -n 8 /srv/output/retry_ir_stdout.log 2>/dev/null
echo "=== [$(date)] 补轮流程结束 ==="
