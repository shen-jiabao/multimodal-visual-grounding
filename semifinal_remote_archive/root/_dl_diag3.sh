#!/bin/bash
# 决定性对照：pip 正在下的确切文件（flashinfer_cubin-0.6.12 正式版），
# 用 curl 从三家镜像各测 15 秒。若都 ~16 MB/s ⇒ 镜像没问题，慢在 pip 自身。
set -u
PKG=flashinfer-cubin
TARGET="flashinfer_cubin-0.6.12-py3-none-any.whl"

for m in "https://mirrors.aliyun.com/pypi/simple" \
         "https://mirrors.ustc.edu.cn/pypi/simple" \
         "https://pypi.tuna.tsinghua.edu.cn/simple"; do
  host=$(echo "$m" | awk -F/ '{print $3}')
  u=$(curl -s --max-time 20 "$m/$PKG/" 2>/dev/null | \
      /root/miniconda3/bin/python3 -c "
import sys, re
from urllib.parse import urljoin
html = sys.stdin.read()
base = '$m/$PKG/'
urls = [urljoin(base, h.split('#')[0]) for h in re.findall(r'href=\"([^\"]+)\"', html)]
c = [x for x in urls if x.endswith('$TARGET')]
print(c[0] if c else '')
")
  if [ -z "$u" ]; then printf "  %-30s ★ 索引里没有 $TARGET\n" "$host"; continue; fi
  printf "  %-30s " "$host"
  curl -sL -o /dev/null --max-time 15 \
    -w "%{speed_download}|%{size_download}|%{time_total}|%{http_code}\n" "$u" 2>/dev/null \
    | awk -F'|' '{printf "%6.2f MB/s  15s 内 %7.2f MB  http=%s\n", $1/1048576, $2/1048576, $4}'
done

echo
echo "=== 对照：同一个文件，pip 自己报的速率（从日志提取）==="
grep -a "flashinfer_cubin" /root/rivermind-data/q38_env.log | tail -3 | tr '\r' '\n' | tail -4 | sed 's/^/  /'
echo "=== DONE ==="
