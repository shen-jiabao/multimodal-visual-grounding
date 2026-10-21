#!/bin/bash
# 查 flash_attention_2 是硬依赖还是可回退
set -u
R=/root/rivermind-data
VLX=$R/vlx-seek
PY=$R/venv_vlx/bin/python

echo "=== 1. 模型 config.json ==="
cat "$VLX/resources/VLX-Seek-1.5-10B/config.json" | sed 's/^/  /'

echo; echo "=== 2. 仓库里 flash_attn / attn_implementation 出现处 ==="
grep -rn "flash_attn\|flash_attention\|attn_implementation" "$VLX" \
     --include=*.py --include=*.json 2>/dev/null | head -30 | sed 's/^/  /'

echo; echo "=== 3. builder.py ==="
cat "$VLX/vlx_seek/models/vlx_seek_1_5/builder.py" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 4. 已装的 attention 相关包 ==="
"$PY" -m pip list 2>/dev/null | grep -iE "flash|attention|triton|xformers" | sed 's/^/  /'
echo "  ---"
"$PY" -c "import torch; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| dev', torch.cuda.get_device_name(0))"

echo; echo "=== 5. 官方 README 对安装的要求 ==="
grep -n -iE "flash|install|torch|transformers" "$VLX/README.md" 2>/dev/null | head -25 | sed 's/^/  /'
echo; echo "=== 6. requirements ==="
for f in requirements.txt requirements*.txt setup.py pyproject.toml; do
  [ -f "$VLX/$f" ] && { echo "  --- $f ---"; cat "$VLX/$f" | sed 's/^/  /'; }
done
echo "=== 结束 ==="
