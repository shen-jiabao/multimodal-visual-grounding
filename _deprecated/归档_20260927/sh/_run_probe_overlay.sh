#!/bin/bash
set +e
echo "===== GPU 占用（应空闲）====="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "===== 用 overlay venv 跑多候选探针 ====="
/root/locate_venv/bin/python /root/_locate_multi_probe.py