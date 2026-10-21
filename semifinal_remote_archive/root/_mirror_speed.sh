#!/bin/bash
# 测各 pip 镜像的**实际下载带宽**（用同一个大文件、固定时间窗对比）。
# 起因：阿里镜像 TTFB 只要 4.5s，但实测单文件吞吐只有 0.4~1 MB/s，
# 后面还要下 torch ~1 GB + nvidia cu13 库 ~3 GB，慢一分钟就是多付一分钟钱。
set -u
. /root/rivermind-data/_q38_env.sh

PKG=numpy
echo "=== 各镜像取 ${PKG} 的 cp312 wheel 并测速（时间窗 20s）==="

for m in "https://mirrors.aliyun.com/pypi/simple" \
         "https://pypi.tuna.tsinghua.edu.cn/simple" \
         "https://mirrors.ustc.edu.cn/pypi/simple" \
         "https://mirrors.cloud.tencent.com/pypi/simple"; do
  host=$(echo "$m" | awk -F/ '{print $3}')
  u=$(curl -s --max-time 20 "$m/$PKG/" 2>/dev/null \
        | grep -oE 'https?://[^"<> ]+\.whl' \
        | grep -E 'cp312|abi3' | grep -E 'manylinux|linux_x86_64' | tail -1)
  if [ -z "$u" ]; then
    printf "  %-38s ★ 拿不到 wheel URL\n" "$host"
    continue
  fi
  # 只测 20 秒，取窗口内下载量算速度（不真下完，省流量）
  r=$(curl -s -o /dev/null --max-time 20 \
        -w '%{speed_download} %{size_download} %{time_total}' "$u" 2>/dev/null)
  spd=$(echo "$r" | awk '{printf "%.2f", $1/1048576}')
  got=$(echo "$r" | awk '{printf "%.1f", $2/1048576}')
  tm=$(echo "$r" | awk '{printf "%.1f", $3}')
  printf "  %-38s %6s MB/s   20s 内下了 %6s MB（实际用时 %ss）\n" \
         "$host" "$spd" "$got" "$tm"
done

echo
echo "=== 当前 pip 配置 ==="
/root/miniconda3/bin/python3 -m pip config list 2>/dev/null | sed 's/^/  /' \
  || tail -4 ~/.pip/pip.conf | sed 's/^/  /'

echo
echo "=== 当前构建进度（磁盘 + 日志尾）==="
du -sh /root/rivermind-data/pipcache 2>/dev/null | sed 's/^/  pipcache /'
tail -3 /root/rivermind-data/q38_env.log | sed 's/^/  /'
echo "=== DONE ==="
