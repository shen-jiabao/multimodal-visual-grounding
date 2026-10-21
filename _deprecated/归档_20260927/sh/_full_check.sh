#!/bin/bash
# 综合检查：模型完整性 + 探针状态 + 进程活性 + 数据上传
MD=/srv/models/LocateAnything-3B
echo "== 模型分片 =="
stat -c "%n %s" $MD/model-0000*.safetensors 2>/dev/null || echo "(未见分片)"
ls $MD/ | grep -i incomplete
echo "== 探针等待日志(尾) =="
tail -n 12 /root/probe_wait.log 2>/dev/null
echo "== 探针运行日志(尾) =="
tail -n 25 /srv/output/probe_stdout.log 2>/dev/null || echo "(探针未开始)"
echo "== 探针结果 =="
ls -la /srv/output/probe_result.json 2>/dev/null || echo "(未生成)"
echo "== 进程 =="
pgrep -af "probe_wait|_probe_locale|snapshot_download|modelscope" | grep -v pgrep | head -6
echo "== 数据上传 =="
ls -la /srv/dataset_rgb.tar 2>/dev/null
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null