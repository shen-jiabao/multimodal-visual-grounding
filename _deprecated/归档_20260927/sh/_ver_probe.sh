#!/bin/bash
set +e
echo "===== 模型目录 requirements / 版本线索 ====="
find /models/LocateAnything-3B -maxdepth 3 -iname "requirements*.txt" -o -iname "*.md" 2>/dev/null | head
echo "--- requirements 内容 ---"
for f in $(find /models/LocateAnything-3B -maxdepth 3 -iname "requirements*.txt" 2>/dev/null | head -3); do
  echo "### $f"; cat "$f"; echo
done
echo "===== 远程建模代码里的版本/attn 线索 ====="
MC=$(find /root/.cache/huggingface/modules -name modeling_locateanything.py 2>/dev/null | head -1)
echo "modeling file: $MC"
grep -nE "transformers|_check_and_adjust_attn|allow_all_kernels|is_torchdtype" "$MC" 2>/dev/null | head -20
echo
echo "===== pip 缓存里是否已有 4.x transformers wheel（可离线装）====="
find /root/.cache/pip -iname "transformers-4*.whl" 2>/dev/null | head
find /root/.cache/pip -iname "*transformers*4*" 2>/dev/null | head
echo
echo "===== 系统 torch 版本（overlay venv 复用）====="
/root/miniconda3/envs/myconda/bin/python -c "import torch,transformers; print('torch',torch.__version__); print('tf',transformers.__version__)" 2>&1 | tail -3
echo
echo "===== 对方 pip 是否还在活动 ====="
ps -ef | grep -E "pip install" | grep -v grep