#!/bin/bash
# 从 ModelScope 并行下载 32B-AWQ 权重（实测单流 30 MB/s；并行可更快）。
# 不用 modelscope 包，直接 curl —— 无额外依赖、可断点续传（-C -）、可核对字节数。
set -u
MID="QuantTrio/Qwen3-VL-32B-Instruct-AWQ"
BASE="https://www.modelscope.cn/models/$MID/resolve/master"
DEST=/root/rivermind-data/models/Qwen3-VL-32B-Instruct-AWQ
LOG=/root/rivermind-data/dl_model.log
PAR=8

mkdir -p "$DEST"
echo "[$(date +%H:%M:%S)] 取文件清单" | tee -a "$LOG"
curl -s --max-time 40 "https://www.modelscope.cn/api/v1/models/$MID/repo/files?Revision=master&Recursive=1" -o /tmp/fl.json || exit 3

python3 - > /tmp/dl_list.txt <<'PYEOF'
import json
d = json.load(open("/tmp/fl.json"))
files = (d.get("Data") or {}).get("Files") or []
for f in files:
    p = f.get("Path") or ""
    s = f.get("Size") or 0
    if p and s > 0:
        print(p + "\t" + str(s))
print("TOTAL\t" + str(sum(f.get("Size") or 0 for f in files)), file=__import__("sys").stderr)
PYEOF
if [ ! -s /tmp/dl_list.txt ]; then echo "★ 文件清单为空" | tee -a "$LOG"; exit 3; fi
tot=$(python3 -c "print(sum(int(l.split(chr(9))[1]) for l in open('/tmp/dl_list.txt') if l.strip()))")
echo "[$(date +%H:%M:%S)] 待下 $(wc -l < /tmp/dl_list.txt) 个文件，合计 $(python3 -c "print(format($tot/1024**3,'.2f'))") GB" | tee -a "$LOG"

dl() {  # $1=相对路径  $2=期望字节
  local rel="$1" want="$2" out="$DEST/$1"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$want" ]; then
    echo "  [跳过] $rel 已完成" >> "$LOG"; return 0
  fi
  curl -sL -C - --retry 5 --retry-delay 3 --max-time 3600 -o "$out" "$BASE/$rel"
  local got; got=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$got" = "$want" ]; then echo "  [完成] $rel $got" >> "$LOG"
  else echo "  ★[不符] $rel 期望 $want 实际 $got" >> "$LOG"; fi
}
export -f dl
export DEST BASE LOG

# 先下小文件（配置/分词器），再并行下大分片
grep -v safetensors /tmp/dl_list.txt | while IFS=$'\t' read -r rel want; do dl "$rel" "$want"; done
grep safetensors /tmp/dl_list.txt | tr '\t' ' ' | xargs -P $PAR -n 2 -I{} bash -c 'dl $0 $1' {}
echo "[$(date +%H:%M:%S)] 全部下载结束" | tee -a "$LOG"

echo "=== 核对 ===" | tee -a "$LOG"
ok=0; bad=0; got_tot=0
while IFS=$'\t' read -r rel want; do
  got=$(stat -c%s "$DEST/$rel" 2>/dev/null || echo 0)
  got_tot=$((got_tot+got))
  if [ "$got" = "$want" ]; then ok=$((ok+1)); else bad=$((bad+1)); echo "  ★不符 $rel 期望 $want 实际 $got"; fi
done < /tmp/dl_list.txt
echo "  字节数正确 $ok 个 | 不符 $bad 个 | 实下 $(python3 -c "print(format($got_tot/1024**3,'.2f'))") GB / 期望 $(python3 -c "print(format($tot/1024**3,'.2f'))") GB" | tee -a "$LOG"
echo "  目录体量: $(du -sh "$DEST" | cut -f1)" | tee -a "$LOG"
df -h /root/rivermind-data | tail -1 | tee -a "$LOG"
[ "$bad" -eq 0 ] && echo "DONE_OK" | tee -a "$LOG" || echo "DONE_BAD" | tee -a "$LOG"