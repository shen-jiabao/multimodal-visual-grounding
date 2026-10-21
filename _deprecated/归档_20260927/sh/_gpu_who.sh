#!/bin/bash
set +e
echo "===== GPU 进程明细 ====="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo
echo "===== 各 GPU 进程的命令行/启动时间 ====="
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  echo "--- PID $pid ---"
  ps -o pid,etime,user,cmd -p "$pid" 2>/dev/null
done
echo
echo "===== pts/0 在跑什么（shell 最近命令无法直接看，列其子进程）====="
ps -ef | grep -v grep | grep -E "python|jupyter" | head -20