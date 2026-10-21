#!/bin/bash
# 复核既有 venv（vllm 0.11.0）是否可用于给 bf16 32B 起服
# 先弄清"要不要重建环境"，再谈开卡——避免白装一遍
set -u
R=/root/rivermind-data
P=$R/venv/bin/python

echo "=== ① 版本 ==="
$P - <<'PY'
import importlib
for m in ("torch", "transformers", "vllm", "flash_attn", "timm", "accelerate"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14s} ✗ {type(e).__name__}: {str(e)[:70]}")
PY

echo; echo "=== ② torch CUDA 链接 ==="
$P -c "import torch;print('  torch',torch.__version__,'| cuda build',torch.version.cuda,'| avail',torch.cuda.is_available(),'| devs',torch.cuda.device_count())" 2>&1 | sed 's/^/  /'

echo; echo "=== ③ vLLM 是否认识 Qwen3-VL ==="
$P - <<'PY'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    names = sorted(ModelRegistry.get_supported_archs())
    hit = [n for n in names if "Qwen3" in n]
    print("  Qwen3 相关架构：")
    for h in hit: print(f"    - {h}")
    if not hit: print("    ★无 ⇒ vLLM 不支持 Qwen3-VL，需升级")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:120]}")
PY

echo; echo "=== ④ 能否实例化 Qwen3-VL 的多模态处理器（不需 GPU、不载权重）==="
$P - <<'PY'
import os
D = "/root/rivermind-data/models/Qwen3-VL-32B-Instruct"
try:
    from transformers import AutoConfig, AutoProcessor
    c = AutoConfig.from_pretrained(D)
    print(f"  config ok: {c.model_type} / architectures={getattr(c,'architectures',None)}")
    p = AutoProcessor.from_pretrained(D)
    print(f"  processor ok: {type(p).__name__}")
    print(f"  image token: {getattr(p,'image_token','?')}")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {str(e)[:150]}")
PY

echo; echo "=== ⑤ 既有起服脚本 ==="
ls -1 $R/_new_serve_vllm.sh $R/_serve*.sh $R/_get_serve*.sh 2>/dev/null | sed 's/^/  /' || echo "  （无）"
echo "  /root/rivermind-data 下与 serve 有关的："
ls -1 $R 2>/dev/null | grep -i -E "serve|vllm" | head -12 | sed 's/^/    /'

echo; echo "=== ⑥ 磁盘明细（起服前确认余量） ==="
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'
df -h $R | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="