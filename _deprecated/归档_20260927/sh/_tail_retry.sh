#!/bin/bash
echo "时间 $(date '+%H:%M:%S')"
echo "----- 日志尾部 -----"
tail -n 14 /root/output/ordinal_retry.log 2>/dev/null
echo "----- checkpoint -----"
python3 -c "import json,os; p='/root/output/locate_ordinal_retry.json'; d=json.load(open(p)) if os.path.exists(p) else {}; print('done:', {k:v.get('n_raw_boxes') for k,v in d.items()})"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader