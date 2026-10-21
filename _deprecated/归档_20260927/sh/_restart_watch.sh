#!/bin/bash
# 重启监视器：加载加固版 _watch_then_launch.sh。
# 用独立脚本承载 kill —— 若直接用 `pkill -f _watch_then_launch.sh`，
# 承载它的那条 bash -c 命令行里也含该串，会把自己一起杀掉。
set -u
echo "== 1) 停掉旧监视器 =="
OLD=$(pgrep -f '_watch_then_launch\.sh')
echo "  旧 PID: ${OLD:-无}"
[ -n "$OLD" ] && kill $OLD 2>/dev/null
sleep 3
echo -n "  停止后残留: "
pgrep -f '_watch_then_launch\.sh' | wc -l

echo
echo "== 2) 语法检查 =="
bash -n /root/_watch_then_launch.sh && echo "  语法 OK"

echo
echo "== 3) 启动新监视器 =="
cd /root
setsid nohup bash /root/_watch_then_launch.sh > /srv/output/watch_outer.log 2>&1 < /dev/null &
sleep 8
echo -n "  新 PID: "; pgrep -f '_watch_then_launch\.sh' | tr '\n' ' '; echo

echo
echo "== 4) 监视日志 =="
cat /srv/output/watch_launch.log
