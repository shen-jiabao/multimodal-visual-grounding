#!/bin/bash
# 停掉平台自启的 vLLM-32B（占显存 38 GB / 49 GB），为 Qwen3-VL-8B 腾空间。
#
# 为什么 pkill 就够：vLLM 由容器入口脚本 /usr/bin/start_all_services.sh
# **一次性**拉起（脚本尾部是 `tail -f /dev/null`，没有守护/重启循环），
# 故杀掉进程后不会被自动拉起。
set -u
echo "=== 停用前 ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader

pkill -f 'vllm serve /root/models/Qwen3-VL-32B-Instruct-AWQ' || true
for _ in $(seq 1 30); do
  pgrep -f 'vllm serve /root/models/Qwen3-VL-32B-Instruct-AWQ' >/dev/null || break
  sleep 2
done
sleep 3

echo "=== 停用后 ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
echo "残留 vllm 进程："
pgrep -af vllm || echo "  none"

echo "=== Python 环境（管线所用）==="
/root/miniconda3/envs/myconda/bin/python - <<'PY'
import torch
import transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda", torch.cuda.is_available())
PY
