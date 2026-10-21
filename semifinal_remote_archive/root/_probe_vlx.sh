#!/bin/bash
# VLX-Seek 落地可行性侦察（只读，不下载）
set -u
R=/root/rivermind-data

echo "=== 磁盘（需容纳 10B 权重 ≈20G + 检测器 + 依赖） ==="
df -h "$R" | tail -1 | sed 's/^/  /'

echo; echo "=== HF 仓库清单 omlab/VLX-Seek-1.5-10B ==="
for base in "https://huggingface.co" "https://hf-mirror.com"; do
  echo "  --- $base ---"
  curl -sS --max-time 25 "$base/api/models/omlab/VLX-Seek-1.5-10B" 2>&1 \
    | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception as e:
    print('    解析失败:', type(e).__name__); raise SystemExit
if 'siblings' not in d:
    print('    无 siblings:', str(d)[:200]); raise SystemExit
fs=[s.get('rfilename','') for s in d['siblings']]
print('    文件数', len(fs))
for f in fs[:12]: print('     ', f)
print('    ...' if len(fs)>12 else '')
print('    architecture:', (d.get('config') or {}).get('architectures'))
print('    tags:', (d.get('tags') or [])[:8])
" 2>&1 | sed 's/^/  /'
  break
done

echo; echo "=== 检测器 WeDetect 是否在 HF（供候选框） ==="
for q in "WeDetect" "omlab"; do
  curl -sS --max-time 25 "https://huggingface.co/api/models?author=omlab&limit=20" 2>&1 \
    | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('    查询失败'); raise SystemExit
for m in d: print('    ', m.get('modelId'), '| downloads', m.get('downloads'))
" 2>&1 | sed 's/^/  /'
  break
done

echo; echo "=== 依赖可用性（仅检查，不安装） ==="
"$R/venv/bin/pip" download --no-deps --dest /tmp/_chk flash-linear-attention 2>&1 | tail -2 | sed 's/^/  fla /'
"$R/venv/bin/pip" index versions causal-conv1d 2>&1 | tail -2 | sed 's/^/  cc1d /'
echo; echo "=== 是否已有 vLLM 注册 VLX-Seek ==="
grep -ril "vlx" "$R/venv/lib/python3"*/site-packages/vllm/model_executor/models/ 2>/dev/null \
  | head -5 | sed 's/^/  /'
echo "  （无输出=未注册，需走自定义 worker）"
echo "=== 结束 ==="
