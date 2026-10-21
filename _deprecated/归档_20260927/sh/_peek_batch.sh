#!/bin/bash
set +e
echo "时间 $(date '+%H:%M:%S')"
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
python3 - <<'PY'
import json,os
p="/root/output/locate_ordinal_batch.json"
if not os.path.exists(p):
    print("checkpoint 尚未生成"); raise SystemExit
d=json.load(open(p))
print("已完成图片:", len(d))
for k,v in d.items():
    if "error" in v:
        print(f"  {k}: ERROR {v['error'][:80]}")
    else:
        print(f"  {k}: raw_boxes={v.get('n_raw_boxes')} core={v.get('core')}")
PY