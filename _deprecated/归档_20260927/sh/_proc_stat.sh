#!/bin/bash
PID=$(pgrep -f _locate_ordinal_batch.py | head -1)
echo "PID=$PID"
for i in 1 2 3; do
  echo "--- sample $i $(date '+%H:%M:%S') ---"
  ps -o pid,etime,time,%cpu,stat,wchan:24 -p "$PID"
  awk '{print "utime="$14" stime="$15" state="$3}' /proc/$PID/stat 2>/dev/null
  sleep 5
done
echo "----- 内核栈（若在 sleep/futex）-----"
cat /proc/$PID/wchan 2>/dev/null; echo
echo "----- 最后内核调用 -----"
cat /proc/$PID/stack 2>/dev/null | head -5 || echo "(无权限读 stack)"
echo "----- checkpoint -----"
python3 -c "import json,os; p='/root/output/locate_ordinal_batch.json'; d=json.load(open(p)) if os.path.exists(p) else {}; print('done:', list(d))"