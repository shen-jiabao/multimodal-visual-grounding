#!/bin/bash
# 核 VLX-Seek 官方接口签名（不占卡）
set -u
VLX=/root/rivermind-data/vlx-seek
cd "$VLX" || exit 1

echo "=== 1. vlx_seek_worker.py 中的类与 def ==="
grep -nE '^(class |    def |def )' vlx_seek_worker.py | sed 's/^/  /'

echo; echo "=== 2. ground 的完整签名 ==="
awk '/def ground/{f=1} f{print "  " $0} f&&/\):/{c++; if(c>=1) exit}' vlx_seek_worker.py | head -20

echo; echo "=== 3. inference.py 中的 def ==="
grep -nE '^(class |def |    def )' inference.py | sed 's/^/  /'

echo; echo "=== 4. load_wedetect_proposals 全文 ==="
awk '/def load_wedetect_proposals/{f=1} f{print "  " $0} f&&/^def [a-z]/&&!/load_wedetect_proposals/{if(++n>1)exit}' inference.py | head -45

echo; echo "=== 5. grounding_single 的模板与回包字段 ==="
grep -n "grounding_single\|result_bbox_list\|object_index\|Detect a single" \
     vlx_seek_worker.py inference.py 2>/dev/null | head -25 | sed 's/^/  /'

echo; echo "=== 6. 冒烟清单 _vlx_smoke20.json ==="
python3 -c "
import json
d=json.load(open('/root/rivermind-data/_vlx_smoke20.json'))
print('  条数',len(d))
print('  前5 ',d[:5])
" 2>&1 | sed 's/^/  /'

echo; echo "=== 7. 权重目录当前状态 ==="
ls -l /root/rivermind-data/vlx-seek/resources/VLX-Seek-1.5-10B/ | sed 's/^/  /'
echo "=== 结束 ==="
