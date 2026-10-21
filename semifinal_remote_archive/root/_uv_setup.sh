#!/bin/bash
# 换成 uv 装包器：
#   ① 停掉慢 pip 构建  ② 用 curl（16 MB/s）取 uv wheel  ③ 装进 conda base（不受 venv 重建影响）
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data
MIR=https://mirrors.aliyun.com/pypi/simple

echo "=== ① 停掉当前 pip 构建 ==="
pkill -f "_q38_build.sh" 2>/dev/null
pkill -f "pip install vllm" 2>/dev/null
sleep 3
if pgrep -af "_q38_build.sh|pip install vllm" >/dev/null 2>&1; then
  echo "  ★ 仍有残留："; pgrep -af "_q38_build.sh|pip install vllm" | sed 's/^/    /'
else
  echo "  ✓ 已全部停止"
fi

echo
echo "=== ② 取 uv wheel（curl 直连镜像）==="
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
if [ -z "$U" ]; then echo "  ★ 索引里没找到 uv wheel"; exit 1; fi
echo "  URL: $U"
curl -sL --max-time 120 -o "$R/tmp/uv.whl" -w "%{size_download}|%{time_total}|%{speed_download}\n" "$U" 2>/dev/null \
  | awk -F'|' '{printf "  下载 %.2f MB，用时 %ss，均速 %.2f MB/s\n", $1/1048576, $2, $3/1048576}'

echo
echo "=== ③ 装 uv 到 conda base（本地 wheel，不走网）==="
/root/miniconda3/bin/python -m pip install --no-index --no-deps -q "$R/tmp/uv.whl" 2>&1 | tail -5
echo "  uv 路径: $(command -v uv || echo '(不在 PATH)')"
[ -x /root/miniconda3/bin/uv ] && echo "  版本: $(/root/miniconda3/bin/uv --version)"

echo
echo "=== ④ 清空 venv_q38（让 uv 重建，避免与 pip 半成品混装）==="
rm -rf "$R/venv_q38"
ls -d "$R/venv_q38" 2>/dev/null && echo "  ★ 删除失败" || echo "  ✓ 已清空"

echo
echo "=== ⑤ 磁盘（旧 pipcache 可清理）==="
du -sh "$R/pipcache" 2>/dev/null | sed 's/^/  旧 pipcache: /'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
