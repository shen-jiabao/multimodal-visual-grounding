#!/bin/bash
# 判定 hf-mirror 是"单连接限速"还是"总额限速"，并查 aria2c 是否可用
set -u
F=model.safetensors
BASE="https://hf-mirror.com/omlab/VLX-Seek-1.5-10B/resolve/main"

echo "=== A. ModelScope 搜索（确认是否镜像了该仓库） ==="
curl -sL --max-time 30 "https://modelscope.cn/api/v1/dolphin/models?PageSize=10&PageNumber=1&SortBy=Default&Name=VLX-Seek" 2>/dev/null \
  | head -c 600 | sed 's/^/  /'
echo

echo; echo "=== B. 4 并发各取 12 MB（测是否单连接限速） ==="
rm -f /tmp/_sp*.bin
for i in 0 1 2 3; do
  st=$((i * 12582912))
  en=$((st + 12582911))
  curl -sL --max-time 40 -r ${st}-${en} -o /tmp/_sp$i.bin \
       -w "%{size_download} %{speed_download}\n" "$BASE/$F" \
       > /tmp/_spw$i.txt 2>/dev/null &
done
wait
tot=0
for i in 0 1 2 3; do
  read -r sz sp < /tmp/_spw$i.txt
  echo "  线程$i  下载 ${sz:-0} 字节  速率 $(awk -v s="${sp:-0}" 'BEGIN{printf "%.2f", s/1048576}') MB/s"
  tot=$(awk -v t="$tot" -v s="${sp:-0}" 'BEGIN{print t+s}')
done
echo "  四线程速率合计 $(awk -v t="$tot" 'BEGIN{printf "%.2f", t/1048576}') MB/s"
rm -f /tmp/_sp*.bin /tmp/_spw*.txt

echo; echo "=== C. 可用工具 ==="
for t in aria2c axel wget curl; do
  command -v $t >/dev/null 2>&1 && echo "  ● $t $(command -v $t)" || echo "  ○ 无 $t"
done

echo; echo "=== D. 现进度 ==="
P=/root/rivermind-data/vlx-seek/resources/VLX-Seek-1.5-10B/model.safetensors
echo "  $(stat -c%s "$P" 2>/dev/null || echo 0) 字节"
echo "=== 结束 ==="
