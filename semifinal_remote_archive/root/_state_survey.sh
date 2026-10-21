#!/bin/bash
# 远端现状盘点：模型 / venv / 核显 / 磁盘（决定要不要重跑、能不能跑 235B）
set -u
R=/root/rivermind-data
echo "=== 1. 顶层目录（含体积） ==="
du -sh "$R"/* 2>/dev/null | sort -h | sed 's/^/  /'

echo; echo "=== 2. 权重点位 ==="
for p in "$R/models" "$R/vlx-seek/resources"; do
  echo "  [$p]"
  ls -1 "$p" 2>/dev/null | head -20 | sed 's/^/    /' || echo "    （不存在）"
done
echo "  --- 全盘搜含 32B / 235B / Qwen 的目录 ---"
find "$R" -maxdepth 4 -type d \( -iname "*32B*" -o -iname "*235B*" -o -iname "*Qwen*" \) \
     2>/dev/null | head -20 | sed 's/^/    /'

echo; echo "=== 3. venv ==="
ls -d "$R"/venv* 2>/dev/null | sed 's/^/  /'
for v in "$R"/venv*/bin/python; do
  [ -x "$v" ] || continue
  echo "  [$v]"
  "$v" -c "
import importlib
for m in ('torch','transformers','vllm'):
    try:
        mod=importlib.import_module(m); print('    ',m,getattr(mod,'__version__','?'))
    except Exception as e:
        print('    ',m,'【缺】')
import torch; print('     cuda',torch.cuda.is_available())
" 2>&1 | sed 's/^/  /'
done

echo; echo "=== 4. GPU 与磁盘 ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
           --format=csv,noheader 2>&1 | sed 's/^/  /'
df -h "$R" | tail -1 | sed 's/^/  /'

echo; echo "=== 5. 是否还有在跑的任务 ==="
pgrep -af "python|vllm|curl" 2>/dev/null | grep -v pgrep | head -10 | cut -c1-100 | sed 's/^/  /'
echo "=== 结束 ==="
