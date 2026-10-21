#!/bin/bash
# 判定瓶颈：网络 还是 数据盘写入？
# 用同一个大文件（torch wheel）分别写 /dev/null、/dev/shm（内存）、数据盘，各 15 秒。
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data

U=$(curl -s --max-time 20 "https://mirrors.aliyun.com/pypi/simple/torch/" 2>/dev/null | \
    /root/miniconda3/bin/python3 -c "
import sys, re
from urllib.parse import urljoin
html = sys.stdin.read()
base = 'https://mirrors.aliyun.com/pypi/simple/torch/'
urls = [urljoin(base, h.split('#')[0]) for h in re.findall(r'href=\"([^\"]+)\"', html)]
c = [x for x in urls if x.endswith('.whl') and 'cp312' in x and 'x86_64' in x]
print(c[-1] if c else '')
")
echo "测试文件: $(basename "$U")"
echo

probe() {  # $1=目标文件描述  $2=输出路径
  printf "  %-22s " "$1"
  curl -sL -o "$2" --max-time 15 -w "%{speed_download}|%{size_download}\n" "$U" 2>/dev/null \
    | awk -F'|' '{printf "%6.2f MB/s  （15s 内 %7.2f MB）\n", $1/1048576, $2/1048576}'
  [ "$2" != "/dev/null" ] && rm -f "$2"
}

echo "=== 同一文件，三种落盘目标（各 15 秒）==="
probe "网络上限(/dev/null)" /dev/null
probe "内存 /dev/shm"      /dev/shm/_spd.bin
probe "数据盘 $R/tmp"      "$R/tmp/_spd.bin"
probe "系统盘 /tmp"        /tmp/_spd.bin

echo
echo "=== 文件系统类型与余量 ==="
df -hT / /tmp /dev/shm "$R" 2>/dev/null | sed 's/^/  /'
echo "  --- 挂载 ---"
mount | grep -E "on / |autodl-tmp|shm" | sed 's/^/  /'

echo
echo "=== 是否有 IO 限速/配额痕迹 ==="
cat /sys/fs/cgroup/io.max 2>/dev/null | sed 's/^/  io.max: /' || echo "  （无 io.max）"
command -v iostat >/dev/null 2>&1 && iostat -dx 2 2 | tail -20 | sed 's/^/  /' || echo "  （无 iostat）"
echo "=== DONE ==="
