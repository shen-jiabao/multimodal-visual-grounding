#!/bin/bash
# 关服前归档：只打包"重要且小"的东西（产物 JSON / 脚本 / 日志），
# 显式排除模型权重、数据集、venv——这些可重新下载/生成，且体积过大。
# 同时输出 SHA256 清单，便于本地核验传输完整性。
set -u
OUT=/tmp/semi_archive.tgz
MAN=/tmp/semi_manifest.txt

echo "===== 1) 体积概览（确认排除项确实是大头）====="
du -sh /srv/output /srv/probe_locate /srv/models /srv/dataset /srv/venv-la2 /srv/venv-locate 2>/dev/null

echo
echo "===== 2) 产物 SHA256 清单 ====="
{
  echo "# 生成时间: $(date -Is)"
  echo "# 主机: $(hostname)"
  echo
  echo "## /srv/output"
  cd /srv/output && sha256sum *.json 2>/dev/null
  echo
  echo "## /srv/probe_locate/out"
  cd /srv/probe_locate/out 2>/dev/null && sha256sum *.json 2>/dev/null
  echo
  echo "## /root 脚本"
  cd /root && sha256sum _*.py _*.sh 2>/dev/null
  echo
  echo "## 日志体积"
  ls -l /srv/output/*.log 2>/dev/null | awk '{print $5, $9}'
} | tee $MAN

echo
echo "===== 3) 打包 ====="
rm -f $OUT
cd /
# 只打包 output / probe_locate / root 下的脚本；不碰 models/dataset/venv
tar -czf $OUT \
  srv/output \
  srv/probe_locate \
  $(ls /root/_*.py /root/_*.sh 2>/dev/null | sed 's#^/##') \
  $MAN
echo "归档: $OUT"
ls -lh $OUT

echo
echo "===== 4) 包内清单（前 60 项）====="
tar -tzf $OUT | wc -l
tar -tzf $OUT | head -60
