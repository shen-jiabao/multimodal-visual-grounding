#!/bin/bash
# VLX-Seek 落地侦察 2：镜像取仓库清单 + GitHub 取官方接口（只读）
set -u
R=/root/rivermind-data

api() {  # $1=base
  for m in "omlab/VLX-Seek-1.5-10B" "omlab/VLX-Seek-10B"; do
    out=$(curl -sS --max-time 25 "$1/api/models/$m" 2>/dev/null)
    if echo "$out" | head -c 1 | grep -q '{'; then
      echo "  [命中] $1 :: $m"
      echo "$out" | python3 -c "
import json,sys
d=json.load(sys.stdin)
fs=[s.get('rfilename','') for s in d.get('siblings',[])]
print('    文件数', len(fs))
for f in fs[:16]: print('     ', f)
print('    architectures:', (d.get('config') or {}).get('architectures'))
print('    license/tags:', (d.get('tags') or [])[:10])
" 2>&1 | sed 's/^/  /'
      return 0
    fi
  done
  echo "  [未命中] $1"
}

echo "=== A. HF 镜像仓库清单 ==="
api "https://hf-mirror.com"

echo; echo "=== B. omlab 名下模型（找检测器） ==="
curl -sS --max-time 25 "https://hf-mirror.com/api/models?author=omlab&limit=40" 2>/dev/null \
  | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception as e:
    print('    查询失败', type(e).__name__); raise SystemExit
print('    共', len(d), '个')
for m in d: print('    ', m.get('modelId'))
" 2>&1 | sed 's/^/  /'

echo; echo "=== C. GitHub 仓库（取官方推理接口与检测器名） ==="
for u in "https://raw.githubusercontent.com/om-ai-lab/VLX-Seek/main/README.md" \
         "https://raw.githubusercontent.com/om-ai-lab/VLX-Seek/master/README.md"; do
  body=$(curl -sSL --max-time 30 "$u" 2>/dev/null)
  if [ -n "$body" ] && [ "${body:0:3}" != "404" ]; then
    echo "  [命中] $u  ($(printf '%s' "$body" | wc -c) 字节)"
    printf '%s' "$body" | grep -inE "wedetect|proposal|worker|VLXSeek|pip install|huggingface.co|region" \
      | head -30 | sed 's/^/    /'
    break
  else
    echo "  [未命中] $u"
  fi
done

echo; echo "=== D. 磁盘与可回收空间 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
du -sh "$R"/models/* 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="
