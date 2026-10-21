#!/bin/bash
# 把 label1024/scripts 下的中文目录复制为 ASCII 名并打包（Windows 解压友好）
cd /root/autodl-tmp/work/label1024/scripts || exit 1
rm -rf /tmp/cn_pack
mkdir -p /tmp/cn_pack
i=0
for d in */; do
  n="${d%/}"
  case "$n" in
    *[!\ -~]*)
      i=$((i+1))
      cp -r "$n" "/tmp/cn_pack/CN_$i"
      echo "  $n -> CN_$i"
      ;;
  esac
done
echo "非 ASCII 目录数：$i"
cd /tmp && tar czf /tmp/scripts_cn.tar.gz cn_pack
ls -la /tmp/scripts_cn.tar.gz
