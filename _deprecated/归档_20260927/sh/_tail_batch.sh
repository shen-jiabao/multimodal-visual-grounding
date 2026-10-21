#!/bin/bash
echo "时间 $(date '+%H:%M:%S')"
echo "----- 日志尾部 -----"
tail -n 20 /root/output/ordinal_batch.log 2>/dev/null
echo "----- checkpoint -----"
python3 -c "import json,os; p='/root/output/locate_ordinal_batch.json'; d=json.load(open(p)) if os.path.exists(p) else {}; print('done:', {k:(v.get('n_raw_boxes'), v.get('core')) for k,v in d.items()})"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader