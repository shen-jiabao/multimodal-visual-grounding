#!/bin/bash
PY=/root/miniconda3/bin/python
echo "=== pip show flash-linear-attention ==="
$PY -m pip show flash-linear-attention 2>&1 | head -8
echo "=== fla 实际是谁 ==="
$PY -c "import fla;print('file',fla.__file__);print('ver',getattr(fla,'__version__','?'))" 2>&1 | tail -4
$PY -c "import fla;print('attrs',[x for x in dir(fla) if not x.startswith('_')][:20])" 2>&1 | tail -2
echo "=== pip list 相关 ==="
$PY -m pip list 2>/dev/null | grep -iE 'flash|fla|einops|triton'
echo "=== 目录 ==="
ls /root/miniconda3/lib/python3.12/site-packages/ | grep -iE 'flash|fla'
