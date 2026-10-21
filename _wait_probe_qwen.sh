#!/bin/bash
# 等 Qwen3-VL 下载完成后自动跑探针（健壮完成判定：无 .incomplete + 4 分片齐全）
LOG=/root/qwen_probe_run.log
exec > >(tee -a $LOG) 2>&1
MD=/srv/models/Qwen3-VL-8B-Instruct
VENV=/srv/venv-locate/bin/python

echo "=== [$(date)] 等 Qwen3-VL 下载完成（健壮判定） ==="
STALL=0
LAST=0
for i in $(seq 1 400); do
  INC=$(ls $MD/*.incomplete 2>/dev/null | wc -l)
  N4=$(ls $MD/model-0000?-of-00004.safetensors 2>/dev/null | wc -l)
  if [ "$INC" -eq 0 ] && [ "$N4" -eq 4 ] && [ -f "$MD/config.json" ] \
     && [ -f "$MD/preprocessor_config.json" ]; then
    TOT=$(du -sb $MD | awk '{print $1}')
    echo "=== 模型完整 (无残留, 4分片齐全, $TOT bytes)，开始探针 ==="
    break
  fi
  CUR=$(du -sb "$MD" 2>/dev/null | awk '{print $1}')
  CUR=${CUR:-0}
  if [ "$CUR" -eq "$LAST" ]; then STALL=$((STALL+1)); else STALL=0; fi
  LAST=$CUR
  echo "[wait] round=$i size=$CUR inc=$INC n4=$N4 stall=$STALL/10 $(date +%T)"
  if [ "$STALL" -ge 10 ]; then
    echo "[$(date)] 停滞 5 分钟，重启续传"
    pkill -f snapshot_download 2>/dev/null || true
    sleep 2
    export MODELSCOPE_DOMAIN=www.modelscope.cn
    /root/miniconda3/envs/myconda/bin/python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/srv/models/Qwen3-VL-8B-Instruct')
" > /root/dl_qwen2.log 2>&1 &
    STALL=0
  fi
  sleep 20
done

echo "=== 运行 Qwen3-VL 探针 ==="
$VENV -u /root/_probe_qwen3vl.py > /srv/output/qwen3vl_probe_stdout.log 2>&1 || {
  echo "探针失败 RC=$?"; tail -n 30 /srv/output/qwen3vl_probe_stdout.log; exit 1; }
echo "=== [$(date)] 探针完成 ==="
tail -n 30 /srv/output/qwen3vl_probe_stdout.log