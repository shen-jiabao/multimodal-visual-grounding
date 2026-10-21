#!/bin/bash
# VLX-Seek 无卡阶段：补齐源码 + 验证依赖 + CPU 侧可导入性
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
VLXVENV=$R/venv_vlx
RES=$VLX/resources

echo "=== 1. 补两个空 __init__.py ==="
for p in "detect_tools/__init__.py" \
         "vlx_seek/models/vlx_seek_1_5/multimodal_encoder/c_radio_v4/__init__.py"; do
  f="$VLX/$p"
  if [ ! -e "$f" ]; then : > "$f"; echo "  已建（空）$p"; else echo "  已存在 $p"; fi
done
n=$(find "$VLX" -type f \( -name '*.py' -o -name '*.txt' -o -name '*.md' \) | wc -l)
echo "  源码/配置文件数 $n"

echo; echo "=== 2a. pip 是否结束 ==="
pgrep -f "_vlx_pip" >/dev/null && echo "  ● 仍在装" || echo "  ○ 已结束"
tail -8 "$R/vlx_pip.log" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 2b. 依赖版本自检（关键：torch 2.10.0 / transformers 5.13.0） ==="
if [ -x "$VLXVENV/bin/python" ]; then
  "$VLXVENV/bin/python" - <<'PY' 2>&1 | tail -12 | sed 's/^/  /'
try:
    import torch, transformers, timm
    print("torch      ", torch.__version__)
    print("transformers", transformers.__version__)
    print("timm       ", timm.__version__)
    print("cuda_available", torch.cuda.is_available())
except Exception as e:
    print("导入失败:", type(e).__name__, e)
PY
else
  echo "  ★ venv 尚未就绪"
fi

echo; echo "=== 3. 下载进度 ==="
pgrep -f "_vlx_dl_model" >/dev/null && echo "  权重 ● 仍在下" || echo "  权重 ○ 已结束"
du -sh "$RES/VLX-Seek-1.5-10B" 2>/dev/null | sed 's/^/  /'
ls -la "$RES/VLX-Seek-1.5-10B" 2>/dev/null | tail -3 | sed 's/^/    /'
pgrep -f "wedetect_base_uni" >/dev/null && echo "  检测器 ● 仍在下" || echo "  检测器 ○ 已结束"
du -sh "$RES/wedetect_base_uni.pth" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 4. WeDetect 调用接口（决定驱动怎么写） ==="
grep -nE "^def |^class |def __init__|argparse|add_argument" \
  "$VLX/detect_tools/WeDetect/generate_proposal.py" 2>/dev/null | head -20 | sed 's/^/  /'

echo; echo "=== 5. worker 是否在模块级即可导入（CPU 不建实例） ==="
if [ -x "$VLXVENV/bin/python" ]; then
  ( cd "$VLX" && "$VLXVENV/bin/python" -c "
import importlib
try:
    m = importlib.import_module('vlx_seek_worker')
    print('  ✓ import vlx_seek_worker 成功；类:', [x for x in dir(m) if 'Worker' in x])
except Exception as e:
    print('  ✗ import 失败:', type(e).__name__, str(e)[:300])
" ) 2>&1 | tail -8 | sed 's/^/  /'
fi

echo; echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
