#!/bin/bash
# 检查数据就绪：上传进度 + 解包 + 图片数 + GPU
echo "== 上传文件 =="
ls -la /srv/dataset_rgb.tar 2>/dev/null || echo "(未上传)"
echo "== 是否已解包 =="
ls -d /srv/dataset/Images/visible 2>/dev/null && echo "已解包" || echo "未解包"
echo "== 图片数校验 =="
N=$(ls /srv/dataset/Images/visible/*.png 2>/dev/null | wc -l)
echo "visible 图片数: $N (应为 2000)"
echo "== 目录结构 =="
ls /srv/dataset/ 2>/dev/null
ls /srv/dataset/queries 2>/dev/null | head
echo "== GPU =="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null
echo "== 磁盘 =="
df -h /srv | tail -1