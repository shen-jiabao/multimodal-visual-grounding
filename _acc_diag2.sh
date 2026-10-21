#!/bin/bash
PY=/root/miniconda3/bin/python
SP=/root/miniconda3/lib/python3.12/site-packages
echo "=== fla 目录 ==="
ls $SP/fla/ | head -30
echo "=== fla.ops ==="
ls $SP/fla/ops/ 2>&1 | head -30
echo "=== 试导入 ==="
$PY -c "import fla.ops;print('fla.ops OK', fla.ops.__path__)" 2>&1 | tail -2
$PY -c "from fla.ops.gated_delta_rule import chunk_gated_delta_rule as f;print('chunk_gated_delta_rule OK')" 2>&1 | tail -2
$PY -c "from fla.ops.common import chunk_gated_delta_rule as f;print('common OK')" 2>&1 | tail -2
echo "=== transformers 是否认账 ==="
$PY -c "from transformers.utils import is_fla_available;print('fla_available',is_fla_available())" 2>&1 | tail -3
$PY -c "from transformers.utils.import_utils import is_flash_linear_attention_available as f;print('flash_linear_attention_available',f())" 2>&1 | tail -3
