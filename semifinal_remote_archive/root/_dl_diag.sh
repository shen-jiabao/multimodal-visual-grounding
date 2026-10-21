#!/bin/bash
# 诊断：pip 究竟在跟谁说话、在下哪个文件、真实速率多少
set -u
. /root/rivermind-data/_q38_env.sh
R=/root/rivermind-data

echo "=== ① pip 进程与它的 TCP 连接（对端 IP 决定走的哪个源）==="
pgrep -af "pip|python" | grep -v "pgrep\|bash -s" | head -8 | sed 's/^/  /'
echo "  --- 连接表 ---"
if command -v ss >/dev/null 2>&1; then
  ss -tnp 2>/dev/null | grep -E "ESTAB" | head -20 | sed 's/^/  /'
else
  netstat -tnp 2>/dev/null | grep ESTAB | head -20 | sed 's/^/  /'
fi
echo "  --- 对端反查 ---"
for ip in $( { ss -tn 2>/dev/null || netstat -tn 2>/dev/null; } | grep ESTAB | awk '{print $5}' | cut -d: -f1 | sort -u | head -6 ); do
  printf "  %-16s " "$ip"
  getent hosts "$ip" 2>/dev/null | head -1 || echo "(无 PTR)"
done

echo
echo "=== ② 真实下载速率（60 秒窗口，只看 pip 的缓存与临时目录）==="
sz() {
  { du -sb "$R/tmp" 2>/dev/null | cut -f1
    du -sb "$R/pipcache" 2>/dev/null | cut -f1
    du -sb /tmp 2>/dev/null | cut -f1
  } | awk '{s+=$1} END{print s+0}'
}
a=$(sz); sleep 60; b=$(sz)
awk -v a="$a" -v b="$b" 'BEGIN{
  d=b-a; printf "  60 秒增长 %.2f MB ⇒ %.2f MB/s\n", d/1048576, d/1048576/60
}'

echo
echo "=== ③ 构建日志原始尾部（40 行，含进度条 \r，看它在哪个包上）==="
tail -c 3000 "$R/q38_env.log" | tr '\r' '\n' | tail -12 | sed 's/^/  /'

echo
echo "=== ④ venv 与磁盘 ==="
du -sh "$R/venv_q38" 2>/dev/null | sed 's/^/  venv: /'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
