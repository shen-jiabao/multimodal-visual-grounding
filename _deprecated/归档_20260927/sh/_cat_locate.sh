#!/bin/bash
# 只读：打印 LocateAnything 推理脚本全文，确认多候选输出接口
for f in /root/locate_run_v3.py /root/locate_run_v2.py /root/locate_run.py; do
  echo "################ $f ################"
  cat "$f" 2>/dev/null
  echo
done