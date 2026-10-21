#!/bin/bash
# 全量核对「非分片文件」（小文件）：大小 + sha256，坏的重下
#   根因：原下载脚本的小文件分支只按 size 跳过、无 sha256 校验 ⇒ 截断文件可留存
#         （实测 tokenizer.json 被截断 → vLLM 起服报 EOF while parsing a value）
#   本脚本：先逐一核对，再对不符者重下（带大小+sha256 双验），最后复验
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B
MODEL=Qwen/Qwen3.8-27B
BASE="https://hf-mirror.com/$MODEL/resolve/main"
OUT=$R/_q38_small.log
: > "$OUT"

{
echo "=== 0) 取官方 manifest ==="
curl -sL --max-time 90 "https://hf-mirror.com/api/models/$MODEL?blobs=true" -o /tmp/_q38m.json 2>/dev/null
python3 - <<'PY' > /tmp/_q38_small.tsv
import json
d = json.load(open("/tmp/_q38m.json", encoding="utf-8"))
for s in d.get("siblings") or []:
    n = s["rfilename"]
    if n.endswith(".safetensors"): continue      # 分片已单独验过
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    print(f"{n}\t{sz}\t{sha}")
PY
echo "非分片文件 $(grep -c . /tmp/_q38_small.tsv) 个"

echo
echo "=== 1) 核对（大小 + sha256） ==="
bad=0; good=0
: > /tmp/_q38_redl.txt
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  f="$D/$name"
  if [ ! -f "$f" ]; then
    echo "✗ 缺失 $name"; echo "$name" >> /tmp/_q38_redl.txt; bad=$((bad+1)); continue
  fi
  s=$(stat -c%s "$f")
  if [ "$s" != "$size" ]; then
    echo "✗ 大小 $name  $s≠$size"; echo "$name" >> /tmp/_q38_redl.txt; bad=$((bad+1)); continue
  fi
  if [ -n "$sha" ]; then
    g=$(sha256sum "$f" | awk '{print $1}')
    if [ "$g" != "$sha" ]; then
      echo "✗ sha256 $name"; echo "$name" >> /tmp/_q38_redl.txt; bad=$((bad+1)); continue
    fi
  fi
  good=$((good+1))
done < /tmp/_q38_small.tsv
echo "⇒ 合格 $good｜不合格 $bad"

echo
echo "=== 2) 重下不合格者（逐个 mkdir 父目录 + 大小/sha256 双验 + 重试） ==="
if [ ! -s /tmp/_q38_redl.txt ]; then
  echo "（无）"
else
  while read -r name; do
    [ -z "$name" ] && continue
    size=$(awk -F'\t' -v n="$name" '$1==n{print $2}' /tmp/_q38_small.tsv)
    sha=$(awk -F'\t' -v n="$name" '$1==n{print $3}' /tmp/_q38_small.tsv)
    mkdir -p "$(dirname "$D/$name")"
    echo "--- 重下 $name（期望 $size 字节）---"
    ok=0
    for t in 1 2 3; do
      rm -f "$D/$name.tmp"
      curl -sSL --connect-timeout 30 --retry 5 --retry-delay 3 --max-time 1200 \
           --speed-limit 20480 --speed-time 30 \
           -o "$D/$name.tmp" "$BASE/$name"
      sz=$(stat -c%s "$D/$name.tmp" 2>/dev/null || echo 0)
      if [ "$sz" != "$size" ]; then
        echo "  第 $t 次：大小 $sz≠$size"; continue
      fi
      if [ -n "$sha" ]; then
        g=$(sha256sum "$D/$name.tmp" | awk '{print $1}')
        if [ "$g" != "$sha" ]; then echo "  第 $t 次：sha256 不符"; continue; fi
      fi
      mv -f "$D/$name.tmp" "$D/$name"; echo "  ✓ 第 $t 次通过（大小+sha256）"; ok=1; break
    done
    [ "$ok" = "0" ] && echo "  ★$name 三次失败"
  done < /tmp/_q38_redl.txt
fi

echo
echo "=== 3) 复验（再核一遍全部非分片文件） ==="
bad2=0; good2=0
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  f="$D/$name"
  [ -f "$f" ] || { echo "✗ 仍缺 $name"; bad2=$((bad2+1)); continue; }
  [ "$(stat -c%s "$f")" = "$size" ] || { echo "✗ 仍大小不符 $name"; bad2=$((bad2+1)); continue; }
  if [ -n "$sha" ] && [ "$(sha256sum "$f" | awk '{print $1}')" != "$sha" ]; then
    echo "✗ 仍 sha 不符 $name"; bad2=$((bad2+1)); continue
  fi
  good2=$((good2+1))
done < /tmp/_q38_small.tsv
echo "⇒ 复验合格 $good2｜仍不合格 $bad2"

echo
echo "=== 4) tokenizer.json 可解析性（JSON 语法） ==="
python3 - <<'PY'
import json
p = "/root/rivermind-data/models/Qwen3.8-27B/tokenizer.json"
try:
    d = json.load(open(p, encoding="utf-8"))
    print(f"  ✓ 可解析；顶层键 {sorted(d.keys())[:6]}｜vocab {len(d.get('model',{}).get('vocab',{}))} 项")
except Exception as e:
    print(f"  ✗ {type(e).__name__}: {e}")
PY
echo "=== DONE ==="
} >> "$OUT" 2>&1