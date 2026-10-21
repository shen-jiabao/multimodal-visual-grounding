#!/bin/bash
# 稳健续传：限速检测自动断开重连，直至字节数与远端一致
set -u
R=/root/rivermind-data
RES=$R/vlx-seek/resources
D=$RES/VLX-Seek-1.5-10B
F=$D/model.safetensors
URL=https://hf-mirror.com/omlab/VLX-Seek-1.5-10B/resolve/main/model.safetensors
mkdir -p "$D"

echo "=== 期望字节数（HEAD） ==="
EXP=$(curl -sIL --max-time 40 "$URL" 2>/dev/null \
      | grep -i '^content-length' | tail -1 | tr -d '\r' | awk '{print $2}')
echo "  EXP=${EXP:-未知}"

echo; echo "=== 启动稳健续传（后台，限速 512B/s×30s 即断开重连） ==="
cat > /tmp/_vlx_dl2.sh <<EOS
#!/bin/bash
F="$F"; URL="$URL"; EXP="${EXP:-0}"; LOG=$R/vlx_dl2.log
try=0
while :; do
  cur=\$(stat -c%s "\$F" 2>/dev/null || echo 0)
  if [ "\$EXP" -gt 0 ] && [ "\$cur" -ge "\$EXP" ]; then
    echo "[\$(date +%H:%M:%S)] DONE \$cur 字节，与远端一致" >> "\$LOG"; break
  fi
  try=\$((try+1))
  echo "[\$(date +%H:%M:%S)] 第 \$try 次续传，起于 \$((cur/1048576)) MB" >> "\$LOG"
  curl -sSL --connect-timeout 30 --retry 3 --retry-delay 3 \
       --speed-limit 512 --speed-time 30 \
       -C - -o "\$F" "\$URL" >> "\$LOG" 2>&1
  rc=\$?
  new=\$(stat -c%s "\$F" 2>/dev/null || echo 0)
  echo "  rc=\$rc，现 \$((new/1048576)) MB" >> "\$LOG"
  [ "\$rc" -ne 0 ] && [ "\$new" -le "\$cur" ] && sleep 5
  [ \$try -gt 200 ] && { echo "  放弃：重试过多" >> "\$LOG"; break; }
done
echo "[dl2] EXIT" >> "\$LOG"
EOS
chmod +x /tmp/_vlx_dl2.sh
pkill -f "VLX-Seek-1.5-10B/resolve" 2>/dev/null && echo "  已停掉停滞的旧 curl" || echo "  （无旧 curl）"
nohup bash /tmp/_vlx_dl2.sh > /dev/null 2>&1 &
echo "  已启动 PID=$!"

sleep 25
echo; echo "=== 25 秒后状态 ==="
tail -4 "$R/vlx_dl2.log" 2>/dev/null | sed 's/^/  /'
s=$(stat -c%s "$F" 2>/dev/null || echo 0); echo "  当前 $((s/1048576)) MB"
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
