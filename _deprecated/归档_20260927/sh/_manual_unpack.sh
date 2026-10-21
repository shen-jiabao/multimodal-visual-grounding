#!/bin/bash
# 手动解包：tar 已上传完整（6.47G），校验并解包到 /srv/dataset
set -e
TAR=/srv/dataset_rgb.tar
echo "== tar 大小 =="
stat -c%s $TAR
echo "== 解包 =="
rm -rf /srv/dataset
mkdir -p /srv/dataset
tar -xf $TAR -C /srv/dataset 2>&1 | head -5 || echo "tar 返回非0"
echo "== 校验图片数 =="
N=$(find /srv/dataset/Images/visible -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' \) 2>/dev/null | wc -l)
echo "visible 图片数: $N (应=2000)"
echo "== 结构 =="
ls /srv/dataset/
ls /srv/dataset/queries/ 2>/dev/null
du -sh /srv/dataset