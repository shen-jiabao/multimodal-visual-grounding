#!/bin/bash
# 下载 Qwen3-VL-8B-Instruct 到 /srv/models/
set -e
MC=/root/miniconda3/envs/myconda/bin/python
export MODELSCOPE_DOMAIN=www.modelscope.cn
echo "=== [$(date)] 开始下载 Qwen3-VL-8B-Instruct ==="
$MC - <<'PY'
from modelscope import snapshot_download
try:
    p = snapshot_download('Qwen/Qwen3-VL-8B-Instruct',
                          local_dir='/srv/models/Qwen3-VL-8B-Instruct')
    print('模型目录:', p)
except Exception as ex:
    print('ERR:', repr(ex))
PY
echo "=== 大小 ==="
du -sh /srv/models/Qwen3-VL-8B-Instruct 2>/dev/null || echo "(未下载)"
ls /srv/models/Qwen3-VL-8B-Instruct 2>/dev/null | head -20
echo "=== [$(date)] 下载结束 ==="