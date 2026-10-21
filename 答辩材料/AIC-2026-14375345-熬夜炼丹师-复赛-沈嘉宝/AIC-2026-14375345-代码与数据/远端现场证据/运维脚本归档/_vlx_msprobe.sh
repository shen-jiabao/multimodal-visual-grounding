#!/bin/bash
# 探 ModelScope 是否有同一仓库 + 实测小段下载速率（不改动现有下载）
set -u
FILE=model.safetensors
MS_BASE="https://modelscope.cn/models/omlab/VLX-Seek-1.5-10B/resolve/master"
HF_BASE="https://hf-mirror.com/omlab/VLX-Seek-1.5-10B/resolve/main"

echo "=== A. ModelScope HEAD ==="
curl -sIL --max-time 40 "$MS_BASE/$FILE" 2>&1 \
  | grep -iE '^(HTTP/|content-length|location)' | tr -d '\r' | sed 's/^/  /' \
  || echo "  请求失败"

echo; echo "=== B. 速率对照（各下 24 MB 到 /dev/null，最多 15 秒） ==="
for tag in MS HF; do
  if [ "$tag" = MS ]; then U="$MS_BASE/$FILE"; else U="$HF_BASE/$FILE"; fi
  out=$(curl -sL --max-time 15 -r 0-25165823 -o /dev/null \
        -w '%{size_download} %{speed_download}' "$U" 2>/dev/null)
  sz=$(echo "$out" | awk '{print $1}')
  sp=$(echo "$out" | awk '{print $2}')
  echo "  $tag  下载 ${sz:-0} 字节  速率 $(awk -v s="${sp:-0}" 'BEGIN{printf "%.2f", s/1048576}') MB/s"
done

echo; echo "=== C. 当前 hf 下载进度（勿动） ==="
F=/root/rivermind-data/vlx-seek/resources/VLX-Seek-1.5-10B/model.safetensors
echo "  $(stat -c%s "$F" 2>/dev/null || echo 0) 字节"
pgrep -f "_vlx_dl2" >/dev/null && echo "  ● 续传循环在跑" || echo "  ○ 循环已退出"
echo "=== 结束 ==="
