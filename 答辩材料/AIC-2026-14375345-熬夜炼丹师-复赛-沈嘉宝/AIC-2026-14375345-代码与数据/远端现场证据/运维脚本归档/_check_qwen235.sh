#!/bin/bash
# 同族替代方案（Qwen3-VL-235B-A22B）的量化版可用性核查 —— 也不需要 GPU
set -u
echo "=== ModelScope 仓库探测 ==="
for repo in \
  "Qwen/Qwen3-VL-235B-A22B-Instruct" \
  "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8" \
  "Qwen/Qwen3-VL-235B-A22B-Instruct-AWQ" \
  "Qwen/Qwen3-VL-30B-A3B-Instruct" \
  "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8" \
  "QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ" \
  "QuantTrio/Qwen3-VL-32B-Instruct-AWQ" \
  "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ" \
  "OpenGVLab/InternVL3-38B-AWQ" \
  "OpenGVLab/InternVL3_5-8B" ; do
  code=$(timeout 18 curl -s -o /tmp/ms.json -w '%{http_code}' \
         "https://modelscope.cn/api/v1/models/$repo" 2>/dev/null || echo 000)
  size=""
  if [ "$code" = "200" ]; then
    size=$(python3 -c "
import json
try:
    d = json.load(open('/tmp/ms.json'))
    v = d.get('Data', {})
    print(f\"下载 {v.get('Downloads', '?')} | 大小 {v.get('Size', '?')}\")
except Exception:
    print('')
" 2>/dev/null)
  fi
  printf "  %-46s HTTP %s  %s\n" "$repo" "$code" "$size"
done

echo
echo "=== ModelScope 搜索：Qwen3-VL 全部（找量化版命名规律）==="
timeout 25 curl -s "https://modelscope.cn/api/v1/models?PageSize=60&PageNumber=1&Name=Qwen3-VL" \
  2>/dev/null | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    items = d.get('Data', {}).get('Models') or d.get('Data') or []
    if isinstance(items, dict): items = items.get('Models', [])
    for m in items[:40]:
        print('   ', m.get('Path') or m.get('Name'))
except Exception as ex:
    print('    解析失败:', type(ex).__name__, str(ex)[:80])
" 2>/dev/null
echo "=== 结束 ==="