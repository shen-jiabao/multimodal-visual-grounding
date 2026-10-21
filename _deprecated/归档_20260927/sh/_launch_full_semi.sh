#!/bin/bash
# 复赛全量启动器：先过闸门，通过后才 setsid nohup 起管线
PY=/root/miniconda3/envs/myconda/bin/python
LOG=/srv/output/qwen_full.log

bash /root/_gate_semi.sh
rc=$?
if [ $rc -ne 0 ]; then echo "闸门未通过（rc=$rc），拒绝启动"; exit 2; fi

if pgrep -f "_qwen_full_pipeline.py" > /dev/null; then
  echo "管线已在运行，PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"
  exit 0
fi

cd /root
setsid nohup $PY -u _qwen_full_pipeline.py > $LOG 2>&1 < /dev/null &
sleep 8
pid=$(pgrep -f _qwen_full_pipeline.py | head -1)
echo "已启动 PID=$pid"
echo "--- 日志首 15 行 ---"
head -n 15 $LOG
