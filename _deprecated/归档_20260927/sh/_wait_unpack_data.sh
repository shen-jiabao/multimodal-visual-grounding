#!/bin/bash
# 等数据集上传完成后自动解包 + 校验；输出日志
set -e
LOG=/root/data_wait.log
exec > >(tee -a $LOG) 2>&1
EXPECT=$((6020*1024*1024))   # dataset_rgb.tar 应约 6.02GB 上限（本地实际 6.02GB）
TAR=/srv/dataset_rgb.tar

echo "=== [$(date)] 等数据上传完成 ==="
for i in $(seq 1 400); do
  SZ=$(stat -c%s "$TAR" 2>/dev/null || echo 0)
  if [ "$SZ" -gt 6000000000 ]; then
    echo "=== 上传完成 ($SZ bytes)，开始解包 ==="
    break
  fi
  echo "[wait] round=$i size=$SZ $(date +%T)"
  sleep 30
done

echo "=== 解包 ==="
rm -rf /srv/dataset
mkdir -p /srv/dataset
tar -xf "$TAR" -C /srv/dataset
echo "=== 校验图片数 ==="
N=$(ls /srv/dataset/Images/visible/*.png 2>/dev/null | wc -l)
echo "visible 图片数: $N (应=2000)"
ls /srv/dataset/queries/mini_queries.json 2>/dev/null && echo "注意: mini_queries 存在(探针包污染?)" || echo "queries 目录正常"
ls /srv/dataset/queries/queries.json 2>/dev/null && echo "queries.json OK" || echo "缺 queries.json!"
ls /srv/dataset/test.json 2>/dev/null && echo "test.json OK" || echo "缺 test.json!"
echo "=== [$(date)] 解包完成 ==="