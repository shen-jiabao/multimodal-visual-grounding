#!/bin/bash
# 交付包准备：① 模型 18 分片全量 SHA256 ② 代码树打包（排除 pycache/outputs）
set -e
cd /root/autodl-tmp/work/code27b || exit 1

echo "=== ① 模型 SHA256（54GB，需要几分钟）==="
sha256sum /root/autodl-tmp/models/Qwen3.8-27B/*.safetensors > /root/model_sha256_27b.txt
wc -l /root/model_sha256_27b.txt

echo "=== ② 代码打包 ==="
tar czf /root/code27b_pkg.tar.gz \
  --exclude='__pycache__' --exclude='outputs' \
  src scripts configs make_sub.py requirements.txt README.md \
  predictions_27b_ref.json
ls -la /root/code27b_pkg.tar.gz

echo "=== ③ 产物副本（annotate 全量预测，供数据归档）==="
cp outputs/predictions_27b_annotate.json /root/predictions_27b_annotate.json
ls -la /root/predictions_27b_annotate.json

echo "=== ④ 环境记录 ==="
{
  echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
  echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
  echo "Python: $(/root/miniconda3/bin/python -V 2>&1)"
  /root/miniconda3/bin/python - <<'PYEOF'
import torch, transformers
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("transformers", transformers.__version__)
PYEOF
  echo "model dir: /root/autodl-tmp/models/Qwen3.8-27B (ModelScope: Qwen/Qwen3.8-27B)"
} > /root/env_record_27b.txt
cat /root/env_record_27b.txt
echo "=== DONE ==="
