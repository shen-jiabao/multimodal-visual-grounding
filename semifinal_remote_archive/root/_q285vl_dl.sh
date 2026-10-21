#!/bin/bash
# Qwen2.5-VL-72B-Instruct-AWQ 下载【跨分片并发 + 边完成边合并 + sha256 校验】
#   移植自已验证的 _q38_dl.sh（同布局同纪律），只换仓库与路径；并**补两处上次的坑**：
#     ① 小文件分支原只按 size 跳过、无校验 ⇒ 上次 tokenizer.json 静默截断
#        本版对每个 .json 做 `json.tool` 解析校验（直接命中该失效模式）
#     ② monitor 的完成判据原按 `model-*-of-*.safetensors` 计数，改为
#        "所有大文件均存在且大小精确相符"（不依赖具体命名模式）
set -u
R=/root/rivermind-data
MODEL=Qwen/Qwen2.5-VL-72B-Instruct-AWQ
BASE="https://hf-mirror.com/$MODEL/resolve/main"
API="https://hf-mirror.com/api/models/$MODEL?blobs=true"
D="$R/models/Qwen2.5-VL-72B-Instruct-AWQ"
TSV=/tmp/_q285_files.tsv
TASKS=/tmp/_q285_tasks.tsv
PARTS="$D/.parts"
CPF=${CPF:-8}
K=${K:-16}
SLIMIT=${SLIMIT:-131072}
STIME=${STIME:-20}

echo "=== 0) 前置 ==="
df -h $R | tail -1 | sed 's/^/  /'
echo "  可用 $(df -BG --output=avail $R | tail -1 | tr -dc '0-9') GB（需 ~41 GB）"
echo "  已下载过？$([ -d "$D" ] && du -sh "$D" | cut -f1 || echo '无')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | sed 's/^/  GPU /'

echo; echo "=== 1) 仓库清单 ==="
curl -sL --max-time 90 "$API" -o /tmp/_q285.json 2>/dev/null
[ -s /tmp/_q285.json ] || { echo "  ★API 无响应，终止"; exit 1; }
python3 - <<'PY' > "$TSV" 2>/tmp/_q285_err || { cat /tmp/_q285_err; exit 1; }
import json
d = json.load(open("/tmp/_q285.json", encoding="utf-8"))
sib = d.get("siblings") or []
if not sib:
    raise SystemExit("no siblings")
tot = 0
for s in sib:
    n = s["rfilename"]
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    tot += sz
    print(f"{n}\t{sz}\t{sha}")
print(f"#TOTAL\t{tot}", file=__import__("sys").stderr)
PY
TOT=$(awk -F'\t' '/^#TOTAL/{print $2}' /tmp/_q285_err)
echo "  文件 $(grep -c . "$TSV") 个，合计 $(( ${TOT:-0} / 1073741824 )) GB"
mkdir -p "$D" "$PARTS"

echo; echo "=== 2) 小文件（含 .json 解析校验）==="
bad=0
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  [ "$size" -gt 1073741824 ] && continue
  out="$D/$name"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
    skip=1
    case "$name" in
      *.json)
        python3 -c "import json,sys;json.load(open(sys.argv[1],encoding='utf-8'))" \
          "$out" >/dev/null 2>&1 || { echo "  ★ $name 大小对但 JSON 解析失败 ⇒ 重下"; skip=0; } ;;
    esac
    [ "$skip" = "1" ] && continue
  fi
  curl -sSL --connect-timeout 30 --retry 5 --retry-delay 3 --max-time 900 \
       -o "$out" "$BASE/$name" >/dev/null 2>&1
  if [ "$(stat -c%s "$out" 2>/dev/null || echo 0)" != "$size" ]; then
    echo "  ✗ $name 大小不符"; bad=$((bad+1)); continue
  fi
  case "$name" in
    *.json) python3 -c "import json,sys;json.load(open(sys.argv[1],encoding='utf-8'))" "$out" >/dev/null 2>&1 \
              || { echo "  ✗ $name JSON 解析失败（截断）"; bad=$((bad+1)); };;
  esac
done < "$TSV"
echo "  小文件完成，异常 $bad 个"
ls -1 "$D" | sed 's/^/    /' | head -30

echo; echo "=== 3) 生成任务池 ==="
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
print(f"#FILES\t{n_file}", file=open("/tmp/_q285_nf","w"))
print(f"#TASKS\t{n_task}", file=open("/tmp/_q285_nt","w"))
PY
NF=$(cut -f2 /tmp/_q285_nf 2>/dev/null); NT=$(cut -f2 /tmp/_q285_nt 2>/dev/null)
echo "  大文件 $NF 个；块任务 ${NT:-0} 个（每片 $CPF 块，$K worker）"

echo; echo "=== 4) 生成 worker / monitor ==="
awk -F'\t' -v k="$K" 'NF>=4 {print > ("/tmp/_q285_wl_" (NR % k))}' "$TASKS"
for i in $(seq 0 $(( K - 1 ))); do : >> "/tmp/_q285_wl_$i"; done

cat > /tmp/_q285_w.sh <<EOS
#!/bin/bash
set -u
BASE="$BASE"; PARTS="$PARTS"; LOG=$R/q285vl_dl.log
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
done < "/tmp/_q285_wl_\$MY"
log "worker 结束"
EOS
chmod +x /tmp/_q285_w.sh

cat > /tmp/_q285_monitor.sh <<EOS
#!/bin/bash
set -u
D="$D"; PARTS="$PARTS"; LOG=$R/q285vl_dl.log; TSV="$TSV"; CPF=$CPF
log() { echo "[\$(date +%H:%M:%S)] mon \$*" >> "\$LOG"; }
for round in \$(seq 1 900); do
  python3 - <<'PY' > /tmp/_q285_ready.txt
import os, glob
D = "$D"; PARTS = "$PARTS"
for line in open("$TSV", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 3: continue
    name, size = p[0], int(p[1])
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
    for i in \$(seq 0 \$(( CPF - 1 ))); do cat "\$PARTS/\$name.part_\$i" >> "\$D/\$name.tmp"; done
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
  done < /tmp/_q285_ready.txt
  done_n=\$(python3 - <<'PY'
import os
D = "$D"; n = 0
for line in open("$TSV", encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) < 2: continue
    name, size = p[0], int(p[1])
    if size <= 1073741824: continue
    f = os.path.join(D, name)
    if os.path.exists(f) and os.path.getsize(f) == size: n += 1
print(n)
PY
)
  [ "\$done_n" -ge "$NF" ] && { log "全部 \$done_n/\$NF 个大文件就位，monitor 退出"; break; }
  sleep 20
done
EOS
chmod +x /tmp/_q285_monitor.sh

echo; echo "=== 5) 启动（monitor + $K worker） ==="
: > "$R/q285vl_dl.log"
nohup bash /tmp/_q285_monitor.sh > /dev/null 2>&1 &
echo "  monitor PID=$!"
for i in $(seq 0 $(( K - 1 ))); do nohup bash /tmp/_q285_w.sh "$i" > /dev/null 2>&1 & done
echo "  $K 个 worker 已启动"

sleep 50
t1=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
sleep 40
t2=$(du -sb "$D" 2>/dev/null | awk '{print $1}')
echo; echo "=== 6) 90 秒后 ==="
echo "  目录 $(( ${t2:-0}/1048576 )) MB / $(( ${TOT:-0}/1048576 )) MB"
awk -v a="${t1:-0}" -v b="${t2:-0}" -v e="${TOT:-0}" 'BEGIN{
  r=(b-a)/40/1048576; printf "  速率 %.2f MB/s\n", r;
  if (r>0) printf "  剩余约 %.0f 分钟\n", (e-b)/1048576/r/60}'
echo "  curl 数 $(pgrep -cf "Qwen2.5-VL-72B-Instruct-AWQ/resolve" || true)"
df -h $R | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="
