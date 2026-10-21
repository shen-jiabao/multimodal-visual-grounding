#!/bin/bash
# Qwen3.8-27B 可行性核查（① 仓库实况 ② 本地 vLLM 是否认识它的架构 ③ 显存/磁盘）
set -u
R=/root/rivermind-data

echo "=== ① HF 仓库实况（Qwen/Qwen3.8-27B） ==="
curl -sL --max-time 90 "https://hf-mirror.com/api/models/Qwen/Qwen3.8-27B?blobs=true" -o /tmp/_q38.json 2>/dev/null
if [ -s /tmp/_q38.json ]; then
  python3 - <<'PY'
import json
d = json.load(open("/tmp/_q38.json", encoding="utf-8"))
sib = d.get("siblings") or []
tot = 0; shards = 0
for s in sib:
    n = s["rfilename"]
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    if n.endswith(".safetensors"):
        shards += 1
    tot += sz
print(f"  文件 {len(sib)}｜safetensors 分片 {shards}｜合计 {tot/1024**3:.2f} GiB")
print(f"  许可证 {d.get('cardData',{}).get('license')}｜下载量 {d.get('downloads')}")
print("  关键文件：", ", ".join(s["rfilename"] for s in sib
      if s["rfilename"] in ("config.json","preprocessor_config.json","model.safetensors.index.json")))
PY
else
  echo "  ★API 无响应"
fi

echo; echo "=== ② 本地 vLLM(0.11.0) 是否登记它的架构 ==="
$R/venv/bin/python - <<'PY'
import json, urllib.request
try:
    d = json.loads(urllib.request.urlopen("file:///tmp/_q38.json").read().decode())
except Exception:
    d = {}
# 从 HF 拿不到 config，先看本地 registry 有没有 qwen3_5 / qwen3.8 相关
from vllm.model_executor.models.registry import ModelRegistry
arch = sorted(ModelRegistry.get_supported_archs())
hit = [a for a in arch if ("Qwen3_5" in a or "Qwen35" in a or "Qwen38" in a
                           or "Qwen3VL" in a or "Qwen3Next" in a)]
print("  registry 里 Qwen 相关：")
for h in hit: print(f"    - {h}")
print(f"  总架构数 {len(arch)}")
PY

echo; echo "=== ③ config.json 的架构名（决定 vLLM 能否加载） ==="
curl -sL --max-time 60 "https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main/config.json" 2>/dev/null \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('  model_type =',d.get('model_type'));print('  architectures =',d.get('architectures'))" 2>&1 \
  | sed 's/^/  /'

echo; echo "=== ④ 本地模型目录与磁盘 ==="
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'
df -h $R | tail -1 | sed 's/^/  /'
echo "  venv vllm 版本: $($R/venv/bin/python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)"
echo "  nvcc（GDN 线性注意力内核编译需要）: $(command -v nvcc || echo '不存在')"
echo "  flash_linear_attention: $($R/venv/bin/python -c 'import fla;print("有")' 2>/dev/null || echo '无')"
echo "  causal_conv1d: $($R/venv/bin/python -c 'import causal_conv1d;print("有")' 2>/dev/null || echo '无')"
echo "=== 结束 ==="