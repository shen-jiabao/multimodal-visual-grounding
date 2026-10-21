#!/bin/bash
# bf16 32B 并发下载【跨分片全局连接池】
#
# 为什么改
#   旧版是"分片串行 × 每片 8 连接"：8 个连接全压在一个文件上，受该文件的服务端吞吐限制
#   （实测 shard1：4.7 GB / 9 min ≈ 8.7 MB/s，即每连接 ~1.1 MB/s）。
#   本版把 14 个分片各切 CPF 块 → 84 个块放进**全局任务池**，K 个 worker 跨分片抢，
#   连接数不再堆在单个文件上。
#
# 磁盘约束（★必须边完成边合并）
#   剩余 ~55 GB 残片 + 55 GB 成品 = 110 GB > 可用 108 GB。
#   故设一个 monitor 进程：某文件所有块齐 ⇒ 立即合并 + 核 sha256 + 删残片，
#   使峰值 ≈ 残片(55) + 一个合并暂存(4.7) ≈ 60 GB。
#
# 速度闸门（旧版踩过的坑）
#   旧版 --speed-limit 1024（1 KB/s）太松，一条 26 KB/s 的烂连接永远不重连。
#   本版 128 KB/s × 20 s。
set -u
R=/root/rivermind-data
MODEL=Qwen/Qwen3-VL-32B-Instruct
BASE="https://hf-mirror.com/$MODEL/resolve/main"
API="https://hf-mirror.com/api/models/$MODEL?blobs=true"
D="$R/models/Qwen3-VL-32B-Instruct"
TSV=/tmp/_q32_files.tsv
TASKS=/tmp/_q32_tasks.tsv
PARTS="$D/.parts"
# ★ CPF=8 是**刻意与旧布局对齐**：旧版每片切 8 块、切法为 ceil(size/8)，
#   与本版完全一致 ⇒ 可把旧残片直接迁移复用（shard 2 已下的 4.1 GB 不丢）。
CPF=${CPF:-8}          # 每文件切块数（与旧版一致，见上）
# 并发数可经 /tmp/_q32_k 外部调（调优用）：实测 16 连接合计仅 3.27 MB/s，
# 而串行版 8 连接同分片达 8.7 MB/s ⇒ 连接数不是越多越好，需实测选点。
K=${K:-$(cat /tmp/_q32_k 2>/dev/null || echo 8)}
SLIMIT=${SLIMIT:-131072}
STIME=${STIME:-20}

echo "=== 0. 停掉旧下载器 ==="
pkill -f "_q32_run.sh" 2>/dev/null && echo "  已停 _q32_run.sh" || echo "  （无旧 runner）"
pkill -f "_q32_w" 2>/dev/null && echo "  已停旧 worker" || true
pkill -f "_q32_monitor" 2>/dev/null && echo "  已停旧 monitor" || true
n=$(pgrep -cf "Qwen3-VL-32B-Instruct/resolve" || true)
[ "${n:-0}" -gt 0 ] && { pkill -f "Qwen3-VL-32B-Instruct/resolve"; echo "  已停 $n 个 curl"; }
sleep 3

echo; echo "=== 1. 仓库清单 ==="
curl -sL --max-time 90 "$API" -o /tmp/_q32.json 2>/dev/null
[ -s /tmp/_q32.json ] || { echo "  ★API 无响应，终止"; exit 1; }
python3 - <<'PY' > "$TSV" 2>/tmp/_q32_err || { cat /tmp/_q32_err; exit 1; }
import json, sys
d = json.load(open("/tmp/_q32.json", encoding="utf-8"))
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
TOT=$(awk -F'\t' '/^#TOTAL/{print $2}' /tmp/_q32_err)
echo "  文件 $(grep -c . "$TSV") 个，合计 $(( ${TOT:-0} / 1073741824 )) GB"
mkdir -p "$D" "$PARTS"

echo; echo "=== 2. 小文件（已完整的跳过） ==="
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  [ "$size" -gt 1073741824 ] && continue
  out="$D/$name"
  [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ] && continue
  curl -sSL --connect-timeout 30 --retry 5 --retry-delay 3 --max-time 900 \
       -o "$out" "$BASE/$name" >/dev/null 2>&1
done < "$TSV"
echo "  完成"

echo; echo "=== 3. 迁移旧残片（旧布局 <name>.parts/part_i → 新 .parts/<name>.part_i） ==="
mig=0
for p in "$D"/*.parts; do
  [ -d "$p" ] || continue
  name=$(basename "$p" .parts)
  for f in "$p"/part_*; do
    [ -f "$f" ] || continue
    case "$f" in *.tmp) rm -f "$f"; continue ;; esac
    idx=${f##*_}
    mv -f "$f" "$PARTS/$name.part_$idx" && mig=$(( mig + 1 ))
  done
  rmdir "$p" 2>/dev/null
done
echo "  迁移 $mig 个残片块（旧 .tmp 已清）"
ls -1 "$PARTS" 2>/dev/null | head -4 | sed 's/^/    /'

echo; echo "=== 3b. 生成全局任务池 ==="
python3 - <<PY > "$TASKS"
import os
D = "$D"; CPF = $CPF
n_file = n_task = 0
for line in open("$TSV", encoding="utf-8"):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 2: continue
    name, size = parts[0], int(parts[1])
    if size <= 1073741824: continue
    n_file += 1
    p = os.path.join(D, name)
    if os.path.exists(p) and os.path.getsize(p) == size:
        continue
    CS = (size + CPF - 1) // CPF
    for i in range(CPF):
        st = i * CS; en = min(st + CS - 1, size - 1)
        if st > en: continue
        print(f"{name}\t{i}\t{st}\t{en}")
        n_task += 1
print(f"#FILES\t{n_file}", file=open("/tmp/_q32_nf","w"))
print(f"#TASKS\t{n_task}", file=open("/tmp/_q32_nt","w"))
PY
NF=$(cut -f2 /tmp/_q32_nf 2>/dev/null); NT=$(cut -f2 /tmp/_q32_nt 2>/dev/null)
echo "  分片总数 $NF；未完成分片的块任务 ${NT:-0} 个（每片 $CPF 块，$K 个 worker）"

echo; echo "=== 4. 生成 worker / monitor ==="
# 静态轮转分派给 K 个 worker（免锁）
awk -F'\t' -v k="$K" 'NF>=4 {print > ("/tmp/_q32_wl_" (NR % k))}' "$TASKS"
for i in $(seq 0 $(( K - 1 ))); do : >> "/tmp/_q32_wl_$i"; done

cat > /tmp/_q32_w.sh <<EOS
#!/bin/bash
set -u
BASE="$BASE"; D="$D"; PARTS="$PARTS"; LOG=$R/q32bf16_par.log
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
done < "/tmp/_q32_wl_\$MY"
log "worker 结束"
EOS
chmod +x /tmp/_q32_w.sh

cat > /tmp/_q32_monitor.sh <<EOS
#!/bin/bash
# 边完成边合并：某文件所有块齐 ⇒ 合并 + 核 sha256 + 删残片
# ★ 计数必须排除 .tmp（正在下载中的临时块）——否则 len(ps) 永远 != CPF，永不合并
set -u
D="$D"; PARTS="$PARTS"; LOG=$R/q32bf16_par.log; TSV="$TSV"
log() { echo "[\$(date +%H:%M:%S)] mon \$*" >> "\$LOG"; }
for round in \$(seq 1 600); do
  python3 - <<'PY' > /tmp/_q32_ready.txt
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
      if [ "\$got" != "\$sha" ]; then log "★\$name sha256 不符（残片或数据坏），删残片待重下"; rm -f "\$D/\$name.tmp"; continue; fi
      log "\$name ✓ 大小+sha256 双验通过（\$(( sz / 1048576 )) MB）"
    else
      log "\$name ✓ 大小通过"
    fi
    mv -f "\$D/\$name.tmp" "\$D/\$name"
    rm -f "\$PARTS/\$name".part_*
  done < /tmp/_q32_ready.txt
  # 全部就位 ⇒ 退出
  done_n=\$(ls -1 "\$D"/model-*-of-00014.safetensors 2>/dev/null | wc -l)
  [ "\$done_n" -ge "$NF" ] && { log "全部 \$done_n 个分片就位，monitor 退出"; break; }
  sleep 20
done
EOS
chmod +x /tmp/_q32_monitor.sh

echo; echo "=== 5. 启动（monitor + $K workers） ==="
nohup bash /tmp/_q32_monitor.sh > /dev/null 2>&1 &
echo "  monitor PID=$!"
for i in $(seq 0 $(( K - 1 ))); do
  nohup bash /tmp/_q32_w.sh "$i" > /dev/null 2>&1 &
done
echo "  $K 个 worker 已启动"

sleep 60
t1=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 45
t2=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
echo; echo "=== 6. 105 秒后 ==="
echo "  目录 $(( ${t2:-0}/1048576 )) MB / $(( ${TOT:-0}/1048576 )) MB"
awk -v a="${t1:-0}" -v b="${t2:-0}" -v e="${TOT:-0}" 'BEGIN{
  r=(b-a)/45/1048576; printf "  速率 %.2f MB/s\n", r;
  if (r>0) printf "  剩余约 %.0f 分钟\n", (e-b)/1048576/r/60}'
echo "  curl 数 $(pgrep -cf 'Qwen3-VL-32B-Instruct/resolve' || true)"
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
