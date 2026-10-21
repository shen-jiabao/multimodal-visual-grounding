#!/bin/bash
# 修正：uv wheel 必须用原名保存，否则 pip 报 "not a valid wheel filename"
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data
MIR=https://mirrors.aliyun.com/pypi/simple

U=$(curl -s --max-time 25 "$MIR/uv/" 2>/dev/null | /root/miniconda3/bin/python3 -c "
import sys, re
from urllib.parse import urljoin
html = sys.stdin.read()
base = '$MIR/uv/'
urls = [urljoin(base, h.split('#')[0]) for h in re.findall(r'href=\"([^\"]+)\"', html)]
c = [x for x in urls if x.endswith('.whl') and 'py3-none-manylinux' in x and 'x86_64' in x and 'aarch64' not in x]
def ver(u):
    m = re.search(r'/uv-([0-9][^-]*)-', u)
    return m.group(1) if m else ''
c.sort(key=lambda x: [int(p) for p in re.findall(r'\d+', ver(x))[:3]])
print(c[-1] if c else '')
")
if [ -z "$U" ]; then echo "★ 索引里没找到 uv wheel"; exit 1; fi
W="$R/tmp/$(basename "$U")"
echo "=== 下载 $(basename "$U") ==="
curl -sL --max-time 120 -o "$W" -w "%{size_download}|%{time_total}|%{speed_download}\n" "$U" 2>/dev/null \
  | awk -F'|' '{printf "  %.2f MB，用时 %ss，均速 %.2f MB/s\n", $1/1048576, $2, $3/1048576}'

echo
echo "=== 装 uv 到 conda base ==="
/root/miniconda3/bin/python -m pip install --no-index --no-deps -q "$W" 2>&1 | tail -4
echo "  uv: $(ls -l /root/miniconda3/bin/uv 2>/dev/null | awk '{print $NF}' || echo '缺失')"
/root/miniconda3/bin/uv --version 2>&1 | sed 's/^/  版本: /'

echo
echo "=== uv 自检：能否走镜像索引 ==="
UV_CACHE_DIR="$R/uvcache" /root/miniconda3/bin/uv pip list --python /root/miniconda3/bin/python 2>&1 | head -3 | sed 's/^/  /'
echo "=== DONE ==="
