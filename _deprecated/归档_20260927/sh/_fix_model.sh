#!/bin/bash
# 官方源装 modelscope(兼容 py3.13) 并下载模型
set -e
MC=/root/miniconda3/envs/myconda/bin/python
echo "== 官方源装 modelscope =="
$MC -m pip install -q --index-url https://pypi.org/simple/ modelscope 2>&1 | tail -3
$MC -c "import modelscope; print('ms ok', modelscope.__version__)" 2>&1 | tail -1
echo "== 下载模型 =="
export MODELSCOPE_DOMAIN=www.modelscope.cn
$MC - <<'PY'
try:
    from modelscope import snapshot_download
    p = snapshot_download('nv-community/LocateAnything-3B',
                          local_dir='/srv/models/LocateAnything-3B')
    print('模型目录:', p)
except Exception as ex:
    print('MODELSCOPE_ERR:', repr(ex))
PY
echo "== 大小 =="
du -sh /srv/models/LocateAnything-3B 2>/dev/null || echo "(未下载成功)"