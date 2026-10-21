#!/bin/bash
# Qwen3.8-27B 0-GPU 可行性核查（**不动现有 venv、不下大权重**）
#   ① 内核能否免编译装上：flash-linear-attention / causal-conv1d 的 wheel 可得性
#   ② transformers 4.57.6 是否认识 model_type=qwen3_5
#   ③ vLLM 是否有支持 Qwen3_5ForConditionalGeneration 的版本
#   ④ 磁盘账（删 AWQ 前后）
set -u
R=/root/rivermind-data
PY=$R/venv/bin/python

echo "=== 0) 现状 ==="
echo "  python $($PY -V 2>&1)｜torch $($PY -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "  transformers $($PY -c 'import transformers;print(transformers.__version__)' 2>/dev/null)"
echo "  nvcc: $(command -v nvcc || echo '不存在')"
df -h $R | tail -1 | sed 's/^/  /'

echo; echo "=== ① 内核 wheel 可得性（不安装，只查询/下载测试）==="
for pkg in flash-linear-attention causal-conv1d; do
  echo "  --- $pkg ---"
  $PY -m pip index versions "$pkg" 2>&1 | head -4 | sed 's/^/    /'
done
echo "  --- 用 pip download 试取 wheel（若只有 sdist，说明需编译 ⇒ 无 nvcc 必失败）---"
mkdir -p /tmp/_whl && cd /tmp/_whl
for pkg in flash-linear-attention causal-conv1d; do
  rm -f *.whl *.tar.gz 2>/dev/null
  out=$($PY -m pip download --no-deps --only-binary=:all: "$pkg" -d /tmp/_whl 2>&1 | tail -3)
  if ls /tmp/_whl/*.whl >/dev/null 2>&1; then
    echo "    ✓ $pkg 有**预编译 wheel**：$(ls /tmp/_whl/*.whl | head -1 | xargs -n1 basename)"
  else
    echo "    ✗ $pkg 无可用 wheel（仅源码 ⇒ 需 nvcc 编译）"
    echo "$out" | head -3 | sed 's/^/        /'
  fi
  rm -f /tmp/_whl/* 2>/dev/null
done

echo; echo "=== ② transformers 4.57.6 是否认识 qwen3_5 ==="
curl -sL --max-time 60 "https://hf-mirror.com/Qwen/Qwen3.8-27B/resolve/main/config.json" \
  -o /tmp/_q38_config.json 2>/dev/null
if [ -s /tmp/_q38_config.json ]; then
  echo "  config.json $(stat -c%s /tmp/_q38_config.json) 字节"
  $PY - <<'PY'
import json, sys
d = json.load(open("/tmp/_q38_config.json", encoding="utf-8"))
print(f"  model_type={d.get('model_type')}  architectures={d.get('architectures')}")
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
    names = list(CONFIG_MAPPING_NAMES.keys())
    hit = [n for n in names if "qwen3" in n.lower()]
    print(f"  transformers 已知 qwen3 系 model_type：{hit}")
    print(f"  {'✓' if d.get('model_type') in names else '✗'} model_type "
          f"'{d.get('model_type')}' {'在' if d.get('model_type') in names else '不在'} CONFIG_MAPPING 里")
except Exception as e:
    print(f"  ✗ 探测失败 {type(e).__name__}: {e}")
PY
else
  echo "  ★config 取不到"
fi

echo; echo "=== ③ vLLM 版本与架构支持 ==="
$PY -m pip index versions vllm 2>&1 | head -3 | sed 's/^/  /'
echo "  --- 本地 vllm registry 里的 Qwen 系 ---"
$PY - <<'PY'
try:
    from vllm.model_executor.models.registry import ModelRegistry
    arch = sorted(ModelRegistry.get_supported_archs())
    print("   ", [a for a in arch if "Qwen" in a])
except Exception as e:
    print(f"    ✗ {e}")
PY
echo "  --- 远端查最新 vllm 是否含 Qwen3_5（GitHub 源码探测）---"
curl -sL --max-time 45 "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/registry.py" \
  2>/dev/null | grep -iE "qwen3_5|qwen35|qwen3\.8" | head -6 | sed 's/^/    /' || echo "    （取不到或未命中）"

echo; echo "=== ④ 磁盘账 ==="
du -sh $R/models/* 2>/dev/null | sed 's/^/  /'
echo "  需要 51.77 GiB ≈ 55.6 GB"
echo "  删 AWQ(20G) 后可用：$(df -BG --output=avail $R | tail -1 | tr -dc '0-9') GB + 20 GB = 约 75 GB"
echo "=== 结束 ==="