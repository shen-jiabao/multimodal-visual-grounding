#!/bin/bash
# 诊断二：pip 慢 20 倍的原因（代理？配置？）
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data

echo "=== ① 代理相关环境变量（若 pip 走代理，就会慢 20 倍）==="
env | grep -i -E "proxy|_PROXY" | sed 's/^/  /' || echo "  （无）"
echo "  --- /etc/network_turbo 是否存在 ---"
ls -l /etc/network_turbo 2>/dev/null | sed 's/^/  /' || echo "  （无此文件）"

echo
echo "=== ② pip 生效配置 ==="
"$R/venv_q38/bin/python" -m pip config list 2>&1 | sed 's/^/  /'
echo "  --- ~/.pip/pip.conf ---"
cat ~/.pip/pip.conf 2>/dev/null | sed 's/^/  /' || echo "  （无）"
echo "  --- 全局 pip.conf ---"
cat /etc/pip.conf 2>/dev/null | sed 's/^/  /' || echo "  （无）"

echo
echo "=== ③ 所有 TCP 状态（含非 ESTAB）==="
( ss -tn 2>/dev/null || netstat -tn 2>/dev/null ) | head -25 | sed 's/^/  /' || echo "  ss/netstat 均不可用"

echo
echo "=== ④ 决定性对照：curl 取 pip 正在下的那个 wheel（15 秒窗口）==="
U=$(curl -s --max-time 20 "https://mirrors.aliyun.com/pypi/simple/flashinfer-cubin/" 2>/dev/null | \
    /root/miniconda3/bin/python3 -c "
import sys, re
from urllib.parse import urljoin
html = sys.stdin.read()
base = 'https://mirrors.aliyun.com/pypi/simple/flashinfer-cubin/'
urls = [urljoin(base, h.split('#')[0]) for h in re.findall(r'href=\"([^\"]+)\"', html)]
c = [x for x in urls if '0.6.12' in x and x.endswith('.whl')]
print(c[-1] if c else '')
")
echo "  URL: ${U:-（取不到）}"
if [ -n "$U" ]; then
  curl -s -o /dev/null --max-time 15 -w "  curl 直连: %{speed_download} B/s  (%{size_download} B 用时 %{time_total}s)\n" "$U" 2>/dev/null \
    | awk '{printf "  curl 直连: %.2f MB/s（15s 内 %.2f MB）\n", $3/1048576, $4/1048576}'
fi

echo
echo "=== ⑤ 当前构建进程仍在否 + 磁盘 ==="
pgrep -af "_q38_build.sh|pip install" | sed 's/^/  /' || echo "  （构建进程已退出）"
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
