#!/bin/bash
# 等模型下载完成后自动跑探针；带停滞检测（5 分钟无增长则重启下载续传）
set -e
LOG=/root/probe_run.log
exec > >(tee -a $LOG) 2>&1
MC=/root/miniconda3/envs/myconda/bin/python

MODEL_DIR=/srv/models/LocateAnything-3B
INC=$MODEL_DIR/model-00001-of-00002.safetensors.incomplete

echo "=== [$(date)] 等待模型下载完成 ==="
STALL_ROUNDS=0   # 连续停滞轮数（每轮探测一次）
for i in $(seq 1 500); do
  # 最终文件已出现且 >1G → 完成
  S1_FINAL=$(stat -c%s "$MODEL_DIR/model-00001-of-00002.safetensors" 2>/dev/null || echo 0)
  S2_FINAL=$(stat -c%s "$MODEL_DIR/model-00002-of-00002.safetensors" 2>/dev/null || echo 0)
  MISSING=""
  for f in config.json model.safetensors.index.json \
           model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
           tokenizer_config.json preprocessor_config.json \
           modeling_locateanything.py processing_locateanything.py; do
    [ -f "$MODEL_DIR/$f" ] || MISSING="$MISSING $f"
  done
  if [ -z "$MISSING" ] && [ "$S1_FINAL" -gt 1000000000 ] && [ "$S2_FINAL" -gt 1000000000 ]; then
    echo "=== 模型完整 (S1=$S1_FINAL S2=$S2_FINAL)，开始探针 ==="
    break
  fi

  # 停滞检测：比较 .incomplete 当前大小与上次
  CUR=$(stat -c%s "$INC" 2>/dev/null || echo 0)
  if [ ! -f "$INC" ] && [ "$S1_FINAL" -le 1000000000 ]; then
    CUR=0
  fi
  CUR=${CUR:-0}
  LAST_INC=${LAST_INC:-0}
  if [ "$CUR" -eq "$LAST_INC" ] && [ "$CUR" -gt 0 ]; then
    STALL_ROUNDS=$((STALL_ROUNDS+1))
    echo "[wait] round=$i 停滞 ${STALL_ROUNDS}/10 (inc=$CUR) $(date +%T)"
  else
    STALL_ROUNDS=0
  fi
  LAST_INC=$CUR

  echo "[wait] round=$i inc=$CUR S1final=$S1_FINAL $(date +%T)"

  if [ "$STALL_ROUNDS" -ge 10 ]; then   # 10×30s = 5 分钟无增长
    echo "[$ (date)] 下载停滞 5 分钟，重启 modelscope 续传"
    # 停掉旧下载，重新 snapshot_download（断点续传）
    pkill -f "snapshot_download" 2>/dev/null || true
    sleep 2
    export MODELSCOPE_DOMAIN=www.modelscope.cn
    setsid nohup $MC -c "
from modelscope import snapshot_download
snapshot_download('nv-community/LocateAnything-3B', local_dir='/srv/models/LocateAnything-3B')
" > /root/model_redl.log 2>&1 < /dev/null &
    STALL_ROUNDS=0
    echo "[$(date)] 已重启下载 PID=$!"
  fi
  sleep 30
done

echo "=== 运行探针 (venv transformers 4.57.1) ==="
/srv/venv-locate/bin/python -u /root/_probe_locale.py > /srv/output/probe_stdout.log 2>&1 || {
  echo "探针失败 RC=$?"; tail -n 30 /srv/output/probe_stdout.log; exit 1
}
echo "=== [$(date)] 探针完成 ==="
tail -n 40 /srv/output/probe_stdout.log