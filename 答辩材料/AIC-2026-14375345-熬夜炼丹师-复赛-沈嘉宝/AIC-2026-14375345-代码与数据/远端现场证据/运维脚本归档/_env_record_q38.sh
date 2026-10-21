#!/bin/bash
# 追加「交付件 0.7963 阶段」的环境实录（不重生成旧记录）
#   为什么不重生成：32B bf16 权重已删，重生成会**丢掉它已记录的分片哈希**；
#   旧记录是"当时那次解析"的产物，本脚本只**追加**新阶段。
set -u
R=/root/rivermind-data
VENV=$R/venv_q38
D=$R/models/Qwen3.8-27B

echo "========== [STAGE-2] 交付件 0.7963 阶段（Qwen3.8-27B / vLLM 0.23.0 / torch 2.11+cu130） =========="
echo "记录时间(UTC): $(date -u '+%Y-%m-%d %H:%M:%S')"

echo; echo "--- 驱动 / CUDA ---"
head -3 /proc/driver/nvidia/version 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader

echo; echo "--- 系统 / Python ---"
. /etc/os-release 2>/dev/null && echo "OS: $PRETTY_NAME"
echo "kernel: $(uname -r)"
"$VENV/bin/python" -c "import sys;print('python', sys.version.split()[0])"

echo; echo "--- 推理栈版本 ---"
"$VENV/bin/python" -c "
import vllm, torch, transformers
print('vllm', vllm.__version__)
print('torch', torch.__version__)
try:
    print('transformers', transformers.__version__)
except Exception as e:
    print('transformers ?', e)
print('torch.version.cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
" 2>&1 | grep -vE "^(WARNING|INFO)"

echo; echo "--- pip freeze（venv_q38）---"
"$VENV/bin/pip" freeze 2>/dev/null

echo; echo "--- 架构登记 ---"
"$VENV/bin/python" -c "
from vllm.model_executor.models.registry import ModelRegistry
a = ModelRegistry.get_supported_archs()
print('架构总数', len(a))
print('Qwen3_5ForConditionalGeneration 已登记:', 'Qwen3_5ForConditionalGeneration' in a)
" 2>&1 | grep -vE "^(WARNING|INFO)"

echo; echo "--- 权重目录 ---"
du -sh "$D" 2>/dev/null
echo "分片数: $(ls -1 "$D"/model-*-of-*.safetensors 2>/dev/null | wc -l)"

echo; echo "--- 权重逐分片 SHA256（由 model.safetensors.index.json 的引用集确定）---"
"$VENV/bin/python" - "$D" <<'PY'
import hashlib, json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
for sh in sorted(set(m.values())):
    f = os.path.join(d, sh)
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    print(f"{h.hexdigest()}  {sh}  {os.path.getsize(f)}")
PY

echo; echo "--- key 文件 SHA256 ---"
sha256sum "$D"/config.json "$D"/tokenizer.json "$D"/model.safetensors.index.json 2>/dev/null

echo; echo "--- 服务命令行（实测）---"
echo "vllm serve $D --served-model-name qwen38-27b --host 0.0.0.0 --port 8000 \\"
echo "  --tensor-parallel-size 2 --trust-remote-code --max-model-len 8192 \\"
echo "  --gpu-memory-utilization 0.90 --max-num-seqs 8 \\"
echo "  --limit-mm-per-prompt {\"image\":1,\"video\":0} --max-num-batched-tokens 2048 \\"
echo "  --disable-custom-all-reduce"
echo "（+ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True；完整脚本见 code/remote_shell/_get_serve_q38.sh）"

echo; echo "--- 客户端环境（本地/远端 HTTP 客户端 venv）---"
$R/venv/bin/python -c "
import openai, PIL
print('openai', openai.__version__)
print('pillow', PIL.__version__)
" 2>&1

echo; echo "--- 数据集 ---"
echo "queries: $(stat -c%s $R/dataset/queries/queries.json 2>/dev/null || echo '?') 字节"
echo "visible 图: $(ls -1 $R/dataset/Images/visible 2>/dev/null | wc -l) 张"

echo; echo "========== [STAGE-2 END] =========="
