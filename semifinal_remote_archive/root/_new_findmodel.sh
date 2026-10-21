#!/bin/bash
# 找出「真实存在、放得下」的量化模型仓库 + 用真实文件测下载速率。
# 约束：盘 49 GB ⇒ bf16 的 32B（65 GB）不可行，必须 4bit 量化（约 18-20 GB）。
check() {  # $1=源前缀  $2=仓库ID
  code=$(curl -s -o /tmp/c.json -w '%{http_code}' --max-time 15 "$1$2")
  extra=""
  if [ "$code" = "200" ]; then
    extra=$(python3 -c "
import json
d=json.load(open('/tmp/c.json'))
for k in ('Name','Id','modelId','id','name'):
    if k in d: extra='Name='+str(d[k]); break
print(extra)
" 2>/dev/null)
  fi
  echo "  $2 -> $code  $extra"
}

echo "=== A) HF 镜像上的候选 ==="
for r in "Qwen/Qwen3-VL-32B-Instruct-AWQ" "Qwen/Qwen3-VL-30B-A3B-Instruct-AWQ" \
         "Qwen/Qwen3-VL-8B-Instruct" "unsloth/Qwen3-VL-32B-Instruct-AWQ" \
         "cyankiwi/Qwen3-VL-32B-Instruct-AWQ" "QuantTrio/Qwen3-VL-32B-Instruct-AWQ" \
         "Qwen/Qwen3-VL-32B-Instruct-GPTQ-Int4" "Qwen/Qwen3-VL-30B-A3B-Instruct"; do
  check "https://hf-mirror.com/api/models/" "$r"
done

echo
echo "=== B) ModelScope 上的候选 ==="
for r in "Qwen/Qwen3-VL-32B-Instruct" "Qwen/Qwen3-VL-30B-A3B-Instruct" \
         "Qwen/Qwen3-VL-8B-Instruct" "Qwen/Qwen3-VL-32B-Instruct-AWQ"; do
  check "https://www.modelscope.cn/api/v1/models/" "$r"
done

echo
echo "=== C) ModelScope 搜索（找量化版真的 ID）==="
curl -s --max-time 25 -X POST "https://www.modelscope.cn/api/v1/dolphin/models" \
  -H 'Content-Type: application/json' \
  -d '{"PageSize":30,"PageNumber":1,"SortBy":"Default","Name":"Qwen3-VL","Criterion":[]}' \
  -o /tmp/s.json
python3 - <<'PYEOF'
import json
try:
    d = json.load(open("/tmp/s.json"))
except Exception as e:
    print("  搜索响应无法解析:", e); raise SystemExit
def walk(o, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("Name", "Path", "ModelId", "Id", "ChineseName") and isinstance(v, str):
                out.append(v)
            walk(v, out)
    elif isinstance(o, list):
        for x in o: walk(x, out)
out = []
walk(d, out)
seen = []
for s in out:
    if s not in seen and ("Qwen3" in s or "VL" in s):
        seen.append(s)
for s in seen[:40]:
    print("   " + s)
print("  共", len(seen), "个候选串")
PYEOF

echo
echo "=== D) 用真实存在的小文件测 ModelScope 速率 ==="
curl -sL --max-time 25 -o /dev/null -w '  config.json 速率 %{speed_download} B/s\n' \
  "https://www.modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct/resolve/master/config.json"

echo
echo "=== E) 用真实存在的权重分片测速率（限 25 秒）==="
curl -sL --max-time 25 -r 0-600000000 -o /dev/null \
  -w '  大分片速率 %{speed_download} B/s  已下 %{size_download} 字节\n' \
  "https://www.modelscope.cn/models/Qwen/Qwen3-VL-32B-Instruct/resolve/master/model-00001-of-00005.safetensors"