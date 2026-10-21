#!/bin/bash
# pip 慢 35 倍的最后一块拼图：是不是解析到了 IPv6（走慢路由）？
set -u
. /root/rivermind-data/_q38_env.sh

U="https://mirrors.aliyun.com/pypi/packages/55/a2/30b20f26f44c2882a43fac7799e712fe3af52421ee25a9c6d66b951a0619/flashinfer_cubin-0.6.12rc3-py3-none-any.whl"

echo "=== ① DNS 解析结果（看有没有 AAAA / IPv6）==="
echo "  --- getent ahosts ---"
getent ahosts mirrors.aliyun.com 2>/dev/null | head -10 | sed 's/^/  /'
echo "  --- ipv4 (A) ---"
getent ahostsv4 mirrors.aliyun.com 2>/dev/null | head -5 | sed 's/^/  /'
echo "  --- ipv6 (AAAA) ---"
getent ahostsv6 mirrors.aliyun.com 2>/dev/null | head -5 | sed 's/^/  /' || echo "  （无 IPv6 记录）"

echo
echo "=== ② curl 强制 IPv4 / IPv6 各测 12 秒 ==="
for fam in 4 6; do
  printf "  -%s  " "$fam"
  out=$(curl -s -$fam -o /dev/null --max-time 12 \
        -w "%{speed_download}|%{size_download}|%{http_code}|%{remote_ip}\n" "$U" 2>/dev/null)
  if [ -z "$out" ]; then
    echo "★ 该地址族不可用（连接失败）"
  else
    echo "$out" | awk -F'|' '{printf "%6.2f MB/s  12s 内 %7.2f MB  http=%s  ip=%s\n", $1/1048576, $2/1048576, $3, $4}'
  fi
done

echo
echo "=== ③ python(urllib3) 对 mirrors.aliyun.com 的解析（pip 用的就是它）==="
/root/miniconda3/bin/python3 - <<'PY'
import socket
host = "mirrors.aliyun.com"
try:
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    seen = []
    for fam, _, _, _, sa in infos:
        name = "IPv6" if fam == socket.AF_INET6 else "IPv4"
        if (name, sa[0]) not in seen:
            seen.append((name, sa[0]))
    for n, a in seen:
        print(f"  {n}  {a}")
    print(f"  默认会优先用: {seen[0][0]} {seen[0][1]}")
except Exception as e:
    print("  解析失败", e)
PY
echo "=== DONE ==="
