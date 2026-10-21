#!/bin/bash
# 8 路分块并行续传 VLX-Seek 权重；取官方 sha256 供校验
set -u
R=/root/rivermind-data
D=$R/vlx-seek/resources/VLX-Seek-1.5-10B
F=$D/model.safetensors
URL=https://hf-mirror.com/omlab/VLX-Seek-1.5-10B/resolve/main/model.safetensors
LOG=$R/vlx_par.log
PARTS=$D/parts
NC=8

echo "=== 1. 取官方元信息（大小 + LFS sha256） ==="
curl -sL --max-time 60 \
  "https://hf-mirror.com/api/models/omlab/VLX-Seek-1.5-10B?blobs=true" \
  -o /tmp/_vlx_api.json 2>/dev/null
RES=$(python3 - <<'PY' 2>/dev/null || echo ""
import json
d = json.load(open("/tmp/_vlx_api.json", encoding="utf-8"))
for s in d.get("siblings", []):
    if s.get("rfilename") == "model.safetensors":
        print(s.get("size", 0), (s.get("lfs") or {}).get("oid", ""))
        break
PY
)
EXP=$(echo "$RES" | awk '{print $1}')
SHA=$(echo "$RES" | awk '{print $2}')
echo "  官方大小   ${EXP:-未知}"
echo "  官方 sha256 ${SHA:-未知}"

echo; echo "=== 2. 停掉单线程续传循环 ==="
pkill -f "_vlx_dl2" 2>/dev/null && echo "  已停循环" || echo "  （无循环）"
pkill -f "VLX-Seek-1.5-10B/resolve" 2>/dev/null && echo "  已停 curl" || echo "  （无 curl）"
sleep 2

echo; echo "=== 3. 现有分片归档（已有字节并入 part_0，不浪费） ==="
mkdir -p "$PARTS"
if [ -f "$F" ]; then
  cur=$(stat -c%s "$F")
  mv -f "$F" "$PARTS/part_0"
  echo "  已把 $cur 字节并入 part_0"
fi
ls -l "$PARTS" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 4. 启动 $NC 路并行（后台） ==="
if [ -z "${EXP:-}" ] || [ "$EXP" -le 0 ] 2>/dev/null; then
  echo "  ★未取到官方大小，不启动（避免拼错文件）"
  exit 1
fi

cat > /tmp/_vlx_par_run.sh <<EOS
#!/bin/bash
set -u
URL="$URL"
OUT="$D/model.safetensors"
PARTS="$PARTS"
LOG="$LOG"
NC=$NC
EXP=$EXP
SHA="$SHA"

log() { echo "[\$(date +%H:%M:%S)] \$*" >> "\$LOG"; }

log "启动 \$NC 路，EXP=\$EXP sha=\${SHA:0:16}"
CS=\$(( (EXP + NC - 1) / NC ))
log "每片 \$CS 字节"

worker() {
  local i=\$1
  local st=\$(( i * CS ))
  local en=\$(( st + CS - 1 ))
  [ "\$en" -ge "\$EXP" ] && en=\$(( EXP - 1 ))
  [ "\$st" -gt "\$en" ] && return 0
  local want=\$(( en - st + 1 ))
  local P="\$PARTS/part_\$i"
  local T="\$PARTS/part_\$i.tmp"
  local tries=0
  while :; do
    local cur=\$(stat -c%s "\$P" 2>/dev/null || echo 0)
    [ "\$cur" -ge "\$want" ] && break
    tries=\$(( tries + 1 ))
    [ "\$tries" -gt 60 ] && { log "part_\$i 重试超限，放弃"; return 1; }
    rm -f "\$T"
    local code=\$(curl -sL --connect-timeout 30 --retry 5 --retry-delay 3 \\
                        --speed-limit 1024 --speed-time 30 \\
                        -r \$(( st + cur ))-\$en -o "\$T" -w '%{http_code}' "\$URL")
    if [ "\$code" = "206" ] && [ -s "\$T" ]; then
      cat "\$T" >> "\$P"
    else
      log "part_\$i 第 \$tries 次非 206（code=\$code），丢弃本次字节"
      sleep 3
    fi
  done
  rm -f "\$T"
  log "part_\$i 完成 \$(stat -c%s "\$P") / \$want 字节"
}

for i in \$(seq 0 \$(( NC - 1 ))); do
  worker \$i &
done

sleep 25
tot=0
for i in \$(seq 0 \$(( NC - 1 ))); do
  s=\$(stat -c%s "\$PARTS/part_\$i" 2>/dev/null || echo 0)
  tot=\$(( tot + s ))
done
log "25 秒时合计 \$(( tot / 1048576 )) MB / \$(( EXP / 1048576 )) MB"

wait
log "全部 worker 退出"

echo "[\$(date +%H:%M:%S)] 合并分片" >> "\$LOG"
# 显式按序合并，避免 part_*.tmp 被 glob 卷进来
: > "\$OUT.tmp"
for i in \$(seq 0 \$(( NC - 1 ))); do
  cat "\$PARTS/part_\$i" >> "\$OUT.tmp"
done
sz=\$(stat -c%s "\$OUT.tmp")
log "合并后 \$sz 字节（期望 \$EXP）"
if [ "\$sz" -ne "\$EXP" ]; then
  log "★大小不符，未替换"
  exit 1
fi
if [ -n "\$SHA" ]; then
  log "校验 sha256（19GB，约 1-2 分钟）"
  got=\$(sha256sum "\$OUT.tmp" | awk '{print \$1}')
  log "  实得 \$got"
  if [ "\$got" != "\$SHA" ]; then
    log "★sha256 不符，未替换"
    exit 1
  fi
  log "  ★sha256 通过"
fi
mv -f "\$OUT.tmp" "\$OUT"
rm -rf "\$PARTS"
log "DONE 权重就位并可验证"
EOS
chmod +x /tmp/_vlx_par_run.sh
nohup bash /tmp/_vlx_par_run.sh > /dev/null 2>&1 &
echo "  PID=$!"

sleep 35
echo; echo "=== 5. 35 秒后 ==="
tail -14 "$LOG" 2>/dev/null | sed 's/^/  /'
tot=0
for p in "$PARTS"/part_*; do
  [ -f "$p" ] || continue
  case "$p" in *.tmp) continue ;; esac
  tot=$(( tot + $(stat -c%s "$p") ))
done
echo "  分片合计 $((tot/1048576)) MB / $((EXP/1048576)) MB"
echo "=== 结束 ==="
