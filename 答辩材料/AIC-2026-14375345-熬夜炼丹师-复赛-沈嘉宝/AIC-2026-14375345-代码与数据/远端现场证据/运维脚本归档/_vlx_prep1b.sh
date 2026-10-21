#!/bin/bash
# VLX-Seek 准备 1b：按需取仓库文件（避开大体积 assets）
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
mkdir -p "$VLX/docs"

echo "=== 1. 仓库文件清单（GitHub API） ==="
tree=$(curl -sSL --max-time 60 --retry 2 \
  "https://api.github.com/repos/om-ai-lab/VLX-Seek/git/trees/main?recursive=1" 2>/dev/null)
if ! echo "$tree" | head -c 1 | grep -q '{'; then
  echo "  API 不通，改用 raw 直接取关键文件"
else
  echo "$tree" | python3 -c "
import json,sys
d=json.load(sys.stdin)
fs=[t['path'] for t in d.get('tree',[]) if t['type']=='blob']
print('  文件总数', len(fs))
keep=[f for f in fs if f.endswith(('.py','.txt','.md','.json','.jinja','.yaml','.yml'))
      and not f.startswith('assets/')]
print('  需要的源码/配置', len(keep), '个：')
for f in keep: print('   ', f)
" 2>&1 | head -60
fi

echo; echo "=== 2. 下载关键文件 ==="
for f in requirements.txt vlx_seek_worker.py docs/inference.md README.md \
         infer.py inference.py main.py demo.py chat_template.jinja; do
  out="$VLX/$f"
  mkdir -p "$(dirname "$out")"
  if curl -sSL --max-time 60 --retry 2 -o "$out" \
       "https://raw.githubusercontent.com/om-ai-lab/VLX-Seek/main/$f" 2>/dev/null \
     && [ -s "$out" ] && ! head -c 20 "$out" | grep -q "404: Not Found"; then
    echo "  ✓ $f  ($(wc -c <"$out") 字节)"
  else
    rm -f "$out"; echo "  ✗ $f"
  fi
done

echo; echo "=== 3. requirements.txt ==="
cat "$VLX/requirements.txt" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 4. worker 对外接口 ==="
grep -nE "^class |^    def |^def " "$VLX/vlx_seek_worker.py" 2>/dev/null | head -30 | sed 's/^/  /'

echo; echo "=== 5. inference.md 关键段 ==="
grep -inE "bbox-list|--task|obj[0-9]|--detector|python |output|json" "$VLX/docs/inference.md" 2>/dev/null \
  | head -35 | sed 's/^/  /'
echo "=== 结束 ==="
