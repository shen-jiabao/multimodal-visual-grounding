#!/bin/bash
# 探测异质候选模型的可得性与体积（走 hf-mirror，只看不下载）
set -u
R=/root/rivermind-data

echo "=== 磁盘 ==="
df -h $R | tail -1 | sed 's/^/  /'
echo "  可用 $(df -BG --output=avail $R | tail -1 | tr -dc '0-9') GB"

echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /' \
  || echo "  （无）"

echo; echo "=== 候选仓库探测（HF API）==="
for m in \
  "Qwen/Qwen2.5-VL-72B-Instruct-AWQ" \
  "Qwen/Qwen2.5-VL-32B-Instruct-AWQ" \
  "OpenGVLab/InternVL3_5-38B" \
  "OpenGVLab/InternVL3_5-38B-AWQ" \
  "OpenGVLab/InternVL3_5-30B-A3B" \
  "OpenGVLab/InternVL3_5-30B-A3B-AWQ" \
  ; do
  j=$(curl -sL --max-time 45 "https://hf-mirror.com/api/models/$m?blobs=true" 2>/dev/null)
  if [ -z "$j" ] || echo "$j" | grep -q '"error"'; then
    echo "  ✗ $m —— 不存在或不可达"
    continue
  fi
  printf '%s' "$j" > /tmp/_hm.json
  python3 - "$m" /tmp/_hm.json <<'PY'
import json, sys
m, p = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as e:
    print(f"  ? {m} —— 解析失败 {e}")
    raise SystemExit(0)
tot = n = 0
for s in d.get("siblings") or []:
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    if sz:
        tot += sz
        n += 1
q = (d.get("config") or {}).get("quantization_config")
print(f"  ✓ {m}")
print(f"      文件 {n} 个｜合计 {tot/1024**3:.1f} GiB"
      f"｜dtype={(d.get('config') or {}).get('torch_dtype')}"
      f"｜quant={q.get('quant_method') if q else '无'}")
PY
done

echo; echo "=== venv_q38 是否支持这些架构 ==="
/root/rivermind-data/venv_q38/bin/python - <<'PY' 2>&1 | tail -8 | sed 's/^/  /'
from vllm.model_executor.models.registry import ModelRegistry
a = ModelRegistry.get_supported_archs()
for x in ("Qwen2_5_VLForConditionalGeneration", "InternVLForConditionalGeneration",
          "InternVLChatModel", "Qwen3VLForConditionalGeneration"):
    print(f"  {x}: {x in a}")
PY
echo "=== 结束 ==="
