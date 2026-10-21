#!/usr/bin/env bash
# 数据集与同尺寸文件排查（只读）
set +e
D=/root/autodl-tmp/rivermind-data

echo "=== 图像数量 ==="
for m in visible infrared depth; do
  n=$(ls $D/dataset/Images/$m 2>/dev/null | wc -l)
  echo "$m: $n"
done

echo
echo "=== 找一个 1683240 字节的文件（提交件主件尺寸） ==="
find /root /autodl-tmp -xdev -type f -size 1683240c 2>/dev/null | head -n 10

echo
echo "=== 全盘搜 predictions_27b / predictions_ref / consensus_pool ==="
find / -xdev -maxdepth 8 \
     \( -path '/proc' -o -path '/sys' -o -path '/root/miniconda3' \
        -o -path '/root/autodl-tmp/rivermind-data/uvcache' \
        -o -path '/root/autodl-tmp/rivermind-data/venv_q38' \) -prune -o \
     \( -name 'predictions*' -o -name 'consensus*' \) -print 2>/dev/null | head -n 20

echo
echo "=== 环境记录 ==="
cat $D/_q38_env.sh 2>/dev/null
echo "--- nvcc/driver ---"
cat /proc/driver/nvidia/version 2>/dev/null | head -n 1

echo
echo "=== _qwen_full_pipeline.py 头部 ==="
head -n 30 $D/_qwen_full_pipeline.py 2>/dev/null

echo
echo "=== done ==="
