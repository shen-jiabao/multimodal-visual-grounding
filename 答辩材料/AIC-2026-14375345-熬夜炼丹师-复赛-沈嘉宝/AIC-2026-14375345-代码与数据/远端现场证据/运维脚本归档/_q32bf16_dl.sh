#!/bin/bash
# 探 Qwen/Qwen3-VL-32B-Instruct（bf16）仓库并启动 8 路并行下载  【修订版】
#
# 修订记录（2026-10-02，实测踩坑）
#   原版把 curl 限速闸设为 --speed-limit 1024（1 KB/s）。实测一条连接以 26 KB/s 爬行
#   （高于闸门 ⇒ 永不重连），7/8 worker 早已完成、第 8 个磨掉 26 分钟只推进 101 MB 中极小部分。
#   修订：① 闸门提到 128 KB/s × 20 s；② 整文件级重试（原版遇大小不符直接 continue，
#   会把该分片**永久跳过**）；③ 启动前先杀掉旧下载器，避免两个进程写同一份残片。
set -u
R=/root/rivermind-data
MODEL=Qwen/Qwen3-VL-32B-Instruct
BASE="https://hf-mirror.com/$MODEL/resolve/main"
API="https://hf-mirror.com/api/models/$MODEL?blobs=true"
D="$R/models/Qwen3-VL-32B-Instruct"
TSV=/tmp/_q32_files.tsv
NC=8
SLIMIT=${SLIMIT:-131072}      # 每连接速率下限：128 KB/s（实测好连接约 1 MB/s）
STIME=${STIME:-20}

echo "=== 0. 停掉旧下载器（避免两进程写同一份残片）==="
if pgrep -f "_q32_run.sh" >/dev/null; then
  pkill -f "_q32_run.sh" && echo "  已停 _q32_run.sh"
else
  echo "  （无旧 runner）"
fi
if pgrep -f "Qwen3-VL-32B-Instruct/resolve" >/dev/null; then
  n=$(pgrep -cf "Qwen3-VL-32B-Instruct/resolve")
  pkill -f "Qwen3-VL-32B-Instruct/resolve" && echo "  已停 $n 个 curl"
else
  echo "  （无旧 curl）"
fi
sleep 3

echo; echo "=== 1. 探测仓库 ==="
curl -sL --max-time 90 "$API" -o /tmp/_q32.json 2>/dev/null
[ -s /tmp/_q32.json ] || { echo "  ★API 无响应，终止"; exit 1; }
python3 - <<'PY' > "$TSV" 2>/tmp/_q32_err || { echo "  解析失败："; cat /tmp/_q32_err; exit 1; }
import json, sys
d = json.load(open("/tmp/_q32.json", encoding="utf-8"))
sib = d.get("siblings") or []
if not sib:
    raise SystemExit("no siblings")
tot = 0
for s in sib:
    name = s["rfilename"]
    size = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    tot += size
    print(f"{name}\t{size}\t{sha}")
print(f"#TOTAL\t{tot}", file=sys.stderr)
PY
TOT=$(awk -F'\t' '/^#TOTAL/{print $2}' /tmp/_q32_err)
echo "  文件数 $(grep -c . "$TSV")；合计 ${TOT:-0} 字节 ≈ $(( ${TOT:-0} / 1073741824 )) GB"
[ "${TOT:-0}" -lt 1000000000 ] && { echo "  ★合计体积异常，终止"; exit 1; }
mkdir -p "$D"

echo; echo "=== 2. 下小文件（已完整的跳过）==="
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  [ "$size" -gt 1073741824 ] && continue
  out="$D/$name"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then echo "  跳过 $name"; continue; fi
  curl -sSL --connect-timeout 30 --retry 5 --retry-delay 3 --max-time 900 \
       -o "$out" "$BASE/$name" && echo "  ✓ $name" || echo "  ✗ $name"
done < "$TSV"

echo; echo "=== 3. 启动大文件并行下载（后台；限速 ${SLIMIT} B/s × ${STIME}s 即重连）==="
cat > /tmp/_q32_run.sh <<EOS
#!/bin/bash
set -u
BASE="$BASE"; D="$D"; TSV="$TSV"; NC=$NC
SLIMIT=$SLIMIT; STIME=$STIME
LOG=$R/q32bf16_dl.log
log() { echo "[\$(date +%H:%M:%S)] \$*" >> "\$LOG"; }
log "=== 启动（限速闸 \${SLIMIT} B/s × \${STIME}s）==="

while IFS=\$'\t' read -r name size sha; do
  [ -z "\$name" ] && continue
  [ "\$size" -gt 1073741824 ] || continue
  OUT="\$D/\$name"; P="\$OUT.parts"
  if [ -f "\$OUT" ] && [ "\$(stat -c%s "\$OUT")" = "\$size" ]; then
    log "跳过 \$name（已完整）"; continue
  fi
  mkdir -p "\$P"
  ftry=0
  while :; do
    ftry=\$(( ftry + 1 ))
    log "开始 \$name（\$(( size / 1048576 )) MB，第 \$ftry 次）"
    CS=\$(( (size + NC - 1) / NC ))
    for i in \$(seq 0 \$(( NC - 1 ))); do
      (
        st=\$(( i * CS )); en=\$(( st + CS - 1 ))
        [ "\$en" -ge "\$size" ] && en=\$(( size - 1 ))
        [ "\$st" -gt "\$en" ] && exit 0
        want=\$(( en - st + 1 ))
        part="\$P/part_\$i"; tmp="\$P/part_\$i.tmp"; tries=0
        while :; do
          cur=\$(stat -c%s "\$part" 2>/dev/null || echo 0)
          [ "\$cur" -ge "\$want" ] && break
          tries=\$(( tries + 1 ))
          [ "\$tries" -gt 80 ] && { log "  part_\$i 重试超限，放弃"; exit 1; }
          rm -f "\$tmp"
          code=\$(curl -sL --connect-timeout 30 --retry 5 --retry-delay 3 \\
                       --speed-limit "\$SLIMIT" --speed-time "\$STIME" \\
                       -r \$(( st + cur ))-\$en -o "\$tmp" -w '%{http_code}' "\$BASE/\$name")
          if [ "\$code" = "206" ] && [ -s "\$tmp" ]; then
            cat "\$tmp" >> "\$part"
          else
            log "  part_\$i 第 \$tries 次非 206（code=\$code）"; sleep 3
          fi
        done
        rm -f "\$tmp"
      ) &
    done
    wait
    : > "\$OUT.tmp"
    for i in \$(seq 0 \$(( NC - 1 ))); do cat "\$P/part_\$i" >> "\$OUT.tmp"; done
    sz=\$(stat -c%s "\$OUT.tmp")
    if [ "\$sz" != "\$size" ]; then
      log "  ★\$name 大小不符 \$sz≠\$size，重试整文件"
      rm -f "\$OUT.tmp"
      [ "\$ftry" -ge 3 ] && { log "  ★\$name 三次失败 ⇒ 跳过（不静默）"; break; }
      continue
    fi
    if [ -n "\$sha" ]; then
      got=\$(sha256sum "\$OUT.tmp" | awk '{print \$1}')
      if [ "\$got" != "\$sha" ]; then
        log "  ★\$name sha256 不符 ⇒ 清残片重下"
        rm -f "\$OUT.tmp"; rm -rf "\$P"; mkdir -p "\$P"
        [ "\$ftry" -ge 3 ] && { log "  ★\$name 三次失败 ⇒ 跳过"; break; }
        continue
      fi
      log "\$name ✓ 大小+sha256 双验通过"
    else
      log "\$name ✓ 大小通过（无官方 sha256）"
    fi
    mv -f "\$OUT.tmp" "\$OUT"; rm -rf "\$P"
    break
  done
done < "\$TSV"
log "DONE 全部大文件处理完毕"
EOS
chmod +x /tmp/_q32_run.sh
nohup bash /tmp/_q32_run.sh > /dev/null 2>&1 &
echo "  PID=$!"

sleep 45
echo; echo "=== 4. 45 秒后 ==="
tail -8 "$R/q32bf16_dl.log" 2>/dev/null | sed 's/^/  /'
t1=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 30
t2=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
echo "  目录 $(( ${t2:-0}/1048576 )) MB / $(( ${TOT:-0}/1048576 )) MB"
awk -v a="${t1:-0}" -v b="${t2:-0}" 'BEGIN{
  printf "  30 秒增量 %.1f MB ⇒ %.2f MB/s\n", (b-a)/1048576, (b-a)/30/1048576}'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
