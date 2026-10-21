#!/bin/bash
# LocateAnything 全量推理启动器：前置校验 + 后台运行 + 日志
set -u
LOG=/root/locate_full_run.log
exec > >(tee -a $LOG) 2>&1
VENV=/srv/venv-locate/bin/python
SCRIPT=/root/_locate_full.py
DATA=/srv/dataset
MODEL=/srv/models/LocateAnything-3B
OUT=/srv/output/locate_bbox_all.json
QUERIES=/srv/dataset/queries/queries.json

echo "=== [$(date)] 全量启动器开始 ==="

# --- 前置校验 ---
echo "== [1/5] 数据校验 =="
NIMG=$(find $DATA/Images/visible -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
echo "visible 图片数: $NIMG (应=2000)"
if [ "$NIMG" -ne 2000 ]; then echo "[ABORT] 图片数 != 2000"; exit 1; fi
[ -f $QUERIES ] && echo "queries.json OK" || { echo "[ABORT] 缺 queries.json"; exit 1; }
[ -f $DATA/test.json ] && echo "test.json OK" || { echo "[ABORT] 缺 test.json"; exit 1; }

echo "== [2/5] 模型校验 =="
for f in config.json model.safetensors.index.json \
         model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  S=$(stat -c%s $MODEL/$f 2>/dev/null || echo 0)
  if [ "$S" -lt 1000000000 ] && [ "$f" != "config.json" ] \
     && [ "$f" != "model.safetensors.index.json" ]; then
    echo "[ABORT] 模型分片异常 $f=$S"; exit 1
  fi
done
echo "模型文件齐全"

echo "== [3/6] GPU 冲突检测 =="
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$')
if [ -n "$PIDS" ]; then
  echo "[ABORT] GPU 已被进程占用: $PIDS"
  for p in $PIDS; do ps -p $p -o pid,cmd --no-headers 2>/dev/null; done
  exit 1
fi
echo "GPU 空闲"

echo "== [4/6] 已有 checkpoint =="
if [ -f $OUT ]; then
  DONE=$($VENV -c "
import json,os
p='$OUT'
d=json.load(open(p)) if os.path.exists(p) else {}
print(len(d))")
  echo "发现 checkpoint: $DONE 条已完成（断点续跑）"
else
  echo "无 checkpoint，从头开始"
fi

echo "== [5/6] 启动全量推理（nohup 后台，50条/checkpoint） =="
cd /root
QUERIES_JSON=$QUERIES OUT_JSON=$OUT \
  setsid nohup nice -n 5 $VENV -u $SCRIPT \
  > /srv/output/locate_full_stdout.log 2>&1 < /dev/null &
PID=$!
echo "PID=$PID"
echo "$PID" > /root/locate_full.pid

echo "== [6/6] 启动确认 =="
sleep 3
if kill -0 $PID 2>/dev/null; then
  echo "进程存活 PID=$PID"
else
  echo "[WARN] 进程未能存活，看日志:"; tail -n 20 /srv/output/locate_full_stdout.log
fi
echo "=== [$(date)] 启动器完成 ==="