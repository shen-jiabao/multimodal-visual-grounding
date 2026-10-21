#!/bin/bash
# Qwen3.8-27B 下载【跨分片并发 + 边完成边合并 + sha256 校验】
#   移植自已验证的 _q32bf16_par.sh（同布局同纪律），只换仓库与路径。
#   教训固化：① 限速闸 128 KB/s×20s（旧版 1 KB/s 太松，烂连接永不重连）
#             ② 整文件级重试（避免大小不符时静默跳过该分片）
#             ③ monitor 计数必须排除 .tmp（否则永不合并）
set -u
R=/root/rivermind-data
MODEL=Qwen/Qwen3.8-27B
BASE="https://hf-mirror.com/$MODEL/resolve/main"
API="https://hf-mirror.com/api/models/$MODEL?blobs=true"
D="$R/models/Qwen3.8-27B"
TSV=/tmp/_q38_files.tsv
TASKS=/tmp/_q38_tasks.tsv
PARTS="$D/.parts"
CPF=${CPF:-8}
K=${K:-16}
SLIMIT=${SLIMIT:-131072}
STIME=${STIME:-20}

echo "=== 0) 驱动版本（决定 cu130 是否可用；无设备节点也能查）==="
echo "  /proc/driver/nvidia/version:"
cat /proc/driver/nvidia/version 2>/dev/null | sed 's/^/    /' || echo "    （不存在）"
echo "  nvidia-smi 头部："
nvidia-smi 2>&1 | head -4 | sed 's/^/    /'
echo "  ★判据：Driver ≥ 580 ⇒ cu130 可用；< 580 ⇒ 需改装 torch cu128"

echo; echo "=== 1) 删 AWQ（用户已明确同意；用途为 H 枚举 + 235B 对照，二者均已封）==="
AWQ="$R/models/Qwen3-VL-32B-Instruct-AWQ"
if [ -d "$AWQ" ]; then
  before=$(du -sm "$AWQ" 2>/dev/null | cut -f1)
  rm -rf "$AWQ" && echo "  已删 $AWQ（${before} MB）"
else
  echo "  （已不存在）"
fi
df -h $R | tail -1 | sed 's/^/  /'

echo; echo "=== 2) 仓库清单 ==="
curl -sL --max-time 90 "$API" -o /tmp/_q38.json 2>/dev/null
[ -s /tmp/_q38.json ] || { echo "  ★API 无响应，终止"; exit 1; }
python3 - <<'PY' > "$TSV" 2>/tmp/_q38_err || { cat /tmp/_q38_err; exit 1; }
import json, sys
d = json.load(open("/tmp/_q38.json", encoding="utf-8"))
sib = d.get("siblings") or []
if not sib: raise SystemExit("no siblings")
tot = 0
for s in sib:
    n = s["rfilename"]
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    tot += sz
    print(f"{n}\t{sz}\t{sha}")
print(f"#TOTAL\t{tot}", file=sys.stderr)
PY
TOT=$(awk -F'\t' '/^#TOTAL/{print $2}' /tmp/_q38_err)
echo "  文件 $(grep -c . "$TSV") 个，合计 $(( ${TOT:-0} / 1073741824 )) GB"
mkdir -p "$D" "$PARTS"

echo; echo "=== 3) 小文件 ==="
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  [ "$size" -gt 1073741824 ] && continue
  out="$D/$name"
  [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ] && continue
  curl -sSL --connect-timeout 30 --retry 5 --retry-delay 3 --max-time 900 \
       -o "$out" "$BASE/$name" >/dev/null 2>&1
done < "$TSV"
echo "  完成（$(ls -1 "$D" 2>/dev/null | wc -l) 个文件在目录）"

echo; echo "=== 4) 生成任务池 ==="
python3 - <<PY > "$TASKS"
import os
D = "$D"; CPF = $CPF
n_file = n_task = 0
for line in open("$TSV", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 2: continue
    name, size = p[0], int(p[1])
    if size <= 1073741824: continue
    n_file += 1
    f = os.path.join(D, name)
    if os.path.exists(f) and os.path.getsize(f) == size: continue
    CS = (size + CPF - 1) // CPF
    for i in range(CPF):
        st = i * CS; en = min(st + CS - 1, size - 1)
        if st > en: continue
        print(f"{name}\t{i}\t{st}\t{en}")
        n_task += 1
print(f"#FILES\t{n_file}", file=open("/tmp/_q38_nf","w"))
print(f"#TASKS\t{n_task}", file=open("/tmp/_q38_nt","w"))
PY
NF=$(cut -f2 /tmp/_q38_nf 2>/dev/null); NT=$(cut -f2 /tmp/_q38_nt 2>/dev/null)
echo "  分片总数 $NF；块任务 ${NT:-0} 个（每片 $CPF 块，$K worker）"

echo; echo "=== 5) 生成 worker / monitor ==="
awk -F'\t' -v k="$K" 'NF>=4 {print > ("/tmp/_q38_wl_" (NR % k))}' "$TASKS"
for i in $(seq 0 $(( K - 1 ))); do : >> "/tmp/_q38_wl_$i"; done

cat > /tmp/_q38_w.sh <<EOS
#!/bin/bash
set -u
BASE="$BASE"; PARTS="$PARTS"; LOG=$R/q38_dl.log
SLIMIT=$SLIMIT; STIME=$STIME
MY=\$1
log() { echo "[\$(date +%H:%M:%S)] w\$MY \$*" >> "\$LOG"; }
while IFS=\$'\t' read -r name idx st en; do
  [ -z "\$name" ] && continue
  want=\$(( en - st + 1 ))
  part="\$PARTS/\$name.part_\$idx"
  cur=\$(stat -c%s "\$part" 2>/dev/null || echo 0)
  [ "\$cur" -ge "\$want" ] && continue
  tries=0
  while :; do
    cur=\$(stat -c%s "\$part" 2>/dev/null || echo 0)
    [ "\$cur" -ge "\$want" ] && break
    tries=\$(( tries + 1 ))
    [ "\$tries" -gt 60 ] && { log "★\$name#\$idx 重试超限"; break; }
    rm -f "\$part.tmp"
    code=\$(curl -sL --connect-timeout 30 --retry 5 --retry-delay 3 \\
                 --speed-limit "\$SLIMIT" --speed-time "\$STIME" \\
                 -r \$(( st + cur ))-\$en -o "\$part.tmp" -w '%{http_code}' "\$BASE/\$name")
    if [ "\$code" = "206" ] && [ -s "\$part.tmp" ]; then
      cat "\$part.tmp" >> "\$part"
    else
      log "\$name#\$idx 第 \$tries 次非 206（code=\$code）"; sleep 3
    fi
  done
  rm -f "\$part.tmp"
done < "/tmp/_q38_wl_\$MY"
log "worker 结束"
EOS
chmod +x /tmp/_q38_w.sh

cat > /tmp/_q38_monitor.sh <<EOS
#!/bin/bash
set -u
D="$D"; PARTS="$PARTS"; LOG=$R/q38_dl.log; TSV="$TSV"
log() { echo "[\$(date +%H:%M:%S)] mon \$*" >> "\$LOG"; }
for round in \$(seq 1 900); do
  python3 - <<'PY' > /tmp/_q38_ready.txt
import os, glob
D = "$D"; PARTS = "$PARTS"
for line in open("$TSV", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3: continue
    name, size, sha = p[0], int(p[1]), p[2]
    if size <= 1073741824: continue
    out = os.path.join(D, name)
    if os.path.exists(out) and os.path.getsize(out) == size: continue
    ps = [x for x in glob.glob(os.path.join(PARTS, name + ".part_*"))
          if not x.endswith(".tmp")]
    if len(ps) != $CPF: continue
    if sum(os.path.getsize(x) for x in ps) != size: continue
    print(name)
PY
  while read -r name; do
    [ -z "\$name" ] && continue
    size=\$(awk -F'\t' -v n="\$name" '\$1==n{print \$2}' "\$TSV")
    sha=\$(awk -F'\t' -v n="\$name" '\$1==n{print \$3}' "\$TSV")
    : > "\$D/\$name.tmp"
    for i in \$(seq 0 \$(( $CPF - 1 ))); do cat "\$PARTS/\$name.part_\$i" >> "\$D/\$name.tmp"; done
    sz=\$(stat -c%s "\$D/\$name.tmp")
    if [ "\$sz" != "\$size" ]; then log "★\$name 大小 \$sz≠\$size，不合并"; rm -f "\$D/\$name.tmp"; continue; fi
    if [ -n "\$sha" ]; then
      got=\$(sha256sum "\$D/\$name.tmp" | awk '{print \$1}')
      if [ "\$got" != "\$sha" ]; then log "★\$name sha256 不符，删残片待重下"; rm -f "\$D/\$name.tmp"; continue; fi
      log "\$name ✓ 大小+sha256 双验通过（\$(( sz / 1048576 )) MB）"
    else
      log "\$name ✓ 大小通过（无官方 sha256）"
    fi
    mv -f "\$D/\$name.tmp" "\$D/\$name"
    rm -f "\$PARTS/\$name".part_*
  done < /tmp/_q38_ready.txt
  done_n=\$(ls -1 "\$D"/model-*-of-*.safetensors 2>/dev/null | wc -l)
  [ "\$done_n" -ge "$NF" ] && { log "全部 \$done_n 个分片就位，monitor 退出"; break; }
  sleep 20
done
EOS
chmod +x /tmp/_q38_monitor.sh

echo; echo "=== 6) 启动（monitor + $K worker） ==="
nohup bash /tmp/_q38_monitor.sh > /dev/null 2>&1 &
echo "  monitor PID=$!"
for i in $(seq 0 $(( K - 1 ))); do nohup bash /tmp/_q38_w.sh "$i" > /dev/null 2>&1 & done
echo "  $K 个 worker 已启动"

sleep 60
t1=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 40
t2=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
echo; echo "=== 7) 100 秒后 ==="
echo "  目录 $(( ${t2:-0}/1048576 )) MB / $(( ${TOT:-0}/1048576 )) MB"
awk -v a="${t1:-0}" -v b="${t2:-0}" -v e="${TOT:-0}" 'BEGIN{
  r=(b-a)/40/1048576; printf "  速率 %.2f MB/s\n", r;
  if (r>0) printf "  剩余约 %.0f 分钟\n", (e-b)/1048576/r/60}'
echo "  curl 数 $(pgrep -cf "Qwen3.8-27B/resolve" || true)"
df -h $R | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="