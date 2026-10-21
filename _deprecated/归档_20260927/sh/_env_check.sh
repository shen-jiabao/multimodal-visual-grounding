#!/bin/bash
set +e
echo "===== pip 进程是否还在 ====="
ps -ef | grep -E "pip install" | grep -v grep
echo
echo "===== myconda transformers 现版本 ====="
/root/miniconda3/envs/myconda/bin/pip show transformers 2>/dev/null | grep -E "^(Name|Version|Location)"
echo
echo "===== base transformers 版本（v3 可能用别的环境）====="
/root/miniconda3/bin/pip show transformers 2>/dev/null | grep -E "^(Name|Version|Location)"
echo
echo "===== conda 环境列表 ====="
ls /root/miniconda3/envs
echo
echo "===== site-packages 里是否残留多个 transformers / 升级半成品 ====="
ls -dt /root/miniconda3/envs/myconda/lib/python3.10/site-packages/transformers* 2>/dev/null
echo
echo "===== 最近的 pip 日志（判断升级是否完成）====="
ls -lt /root/.cache/pip/log 2>/dev/null | head