#!/bin/bash
# InternVL3.5-38B 前置侦察：机器状态 / 磁盘余量 / 上次成果是否幸存 / vLLM 是否支持
set -u
echo "=== 主机 / 时间 ==="; hostname; date
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader 2>/dev/null || echo "  无 nvidia-smi"
echo; echo "=== 磁盘（全部挂载） ==="
df -h 2>/dev/null | grep -Ev 'tmpfs|udev' | head -14
echo; echo "=== 上次成果是否幸存 ==="
for p in /root/rivermind-data \
         /root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ \
         /root/rivermind-data/venv \
         /root/rivermind-data/dataset \
         /root/rivermind-data/_qwen_full_pipeline.py ; do
  if [ -e "$p" ]; then echo "  [有] $p  ($(du -sh "$p" 2>/dev/null | cut -f1))"
  else echo "  [无] $p"; fi
done
echo; echo "=== 数据集张数 ==="
D=/root/rivermind-data/dataset
for m in visible infrared depth; do
  echo "  Images/$m : $(ls "$D/Images/$m" 2>/dev/null | wc -l) 张"
done
echo; echo "=== venv 关键包 ==="
V=/root/rivermind-data/venv/bin/python
if [ -x "$V" ]; then
  $V -c "import sys;print('  python', sys.version.split()[0])"
  for m in vllm transformers torch openai PIL numpy; do
    $V -c "import $m;print('  $m', getattr($m,'__version__','?'))" 2>/dev/null \
      || echo "  $m 缺失"
  done
else echo "  无 venv"; fi
echo; echo "=== vLLM 是否登记了 InternVL ==="
$V - <<'PYEOF' 2>/dev/null || echo "  (查询失败)"
import os, sys
try:
    from vllm.model_executor.models.registry import ModelRegistry
    names = [k for k in ModelRegistry.get_supported_archs()
             if "Intern" in k or "intern" in k]
    print("  注册的 InternVL 架构:", names if names else "★无")
except Exception as ex:
    print("  查询失败:", type(ex).__name__, ex)
PYEOF
echo; echo "=== 可写性探测 ==="
for d in /root/rivermind-data /data/tini /data; do
  if [ -d "$d" ]; then
    if touch "$d/.wtest" 2>/dev/null; then echo "  [可写] $d"; rm -f "$d/.wtest"
    else echo "  [只读/不可写] $d"; fi
  else echo "  [不存在] $d"; fi
done
echo; echo "=== 网络 ==="
timeout 12 curl -sI https://www.modelscope.cn 2>/dev/null | head -1 || echo "  modelscope 不可达"
timeout 12 curl -sI https://huggingface.co 2>/dev/null | head -1 || echo "  HF 不可达"
echo "=== 侦察结束 ==="