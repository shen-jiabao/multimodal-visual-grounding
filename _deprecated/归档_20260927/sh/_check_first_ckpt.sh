#!/bin/bash
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
echo "=== proc ==="
pgrep -af _locate_full.py
echo "=== output dir ==="
ls -la /srv/output/ 2>/dev/null
echo "=== checkpoint ==="
for f in /srv/output/locate_full_checkpoint.json /srv/output/checkpoint.json; do
  [ -f "$f" ] && echo "found $f" && head -c 300 "$f" && echo
done
echo "=== log tail ==="
tail -n 8 /srv/output/locate_full_stdout.log 2>/dev/null || tail -n 8 /srv/output/locate_full.log 2>/dev/null
echo "=== json entries ==="
python3 -c "import json,glob
for p in glob.glob('/srv/output/*.json'):
    try:
        d=json.load(open(p)); print(p, 'entries=', len(d) if isinstance(d,(list,dict)) else '?')
    except Exception as e: print(p,'ERR',e)" 2>/dev/null