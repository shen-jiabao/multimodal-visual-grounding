#!/bin/bash
# 取驱动所需的全部接口细节
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek

echo "=== A. inference.py 是否可安全 import（有无 __main__ 守卫） ==="
grep -nE '^if __name__|^def main|^main\(\)' "$VLX/inference.py" | sed 's/^/  /'

echo; echo "=== B. load_wedetect_proposals 全文（官方候选生成） ==="
awk 'NR>=155 && NR<=205' "$VLX/inference.py" | sed 's/^/  /'

echo; echo "=== C. SimpleYOLOWorldDetector 构造与调用 ==="
awk 'NR>=1047 && NR<=1075' "$VLX/detect_tools/WeDetect/generate_proposal.py" | sed 's/^/  /'
grep -nE "    def (forward|__call__|predict|detect|__init__)" \
  "$VLX/detect_tools/WeDetect/generate_proposal.py" | head -12 | sed 's/^/  /'

echo; echo "=== D. worker.__init__（加载什么、默认 device） ==="
awk 'NR>=36 && NR<=72' "$VLX/vlx_seek_worker.py" | sed 's/^/  /'

echo; echo "=== E. _validate_boxes（坐标口径） ==="
awk 'NR>=53 && NR<=80' "$VLX/vlx_seek_worker.py" | sed 's/^/  /'

echo; echo "=== F. 下载进度 ==="
s=$(stat -c%s "$R/vlx-seek/resources/VLX-Seek-1.5-10B/model.safetensors" 2>/dev/null || echo 0)
echo "  $((s/1048576)) MB / 19109 MB（$(python3 -c "print(f'{$s/20033961788*100:.1f}%')")）"
tail -3 "$R/vlx_dl2.log" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="
