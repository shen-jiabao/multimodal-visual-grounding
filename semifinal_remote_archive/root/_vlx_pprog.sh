#!/bin/bash
# 并行下载实时进度（含 .tmp 暂存）+ API 结构排查
set -u
D=/root/rivermind-data/vlx-seek/resources/VLX-Seek-1.5-10B
PARTS=$D/parts
LOG=/root/rivermind-data/vlx_par.log
EXP=20033961788

echo "=== 1. 各分片（含暂存） ==="
for i in $(seq 0 7); do
  a=$(stat -c%s "$PARTS/part_$i" 2>/dev/null || echo 0)
  b=$(stat -c%s "$PARTS/part_$i.tmp" 2>/dev/null || echo 0)
  printf "  part_%s  已并 %6d MB  暂存 %6d MB\n" "$i" $((a/1048576)) $((b/1048576))
done

echo; echo "=== 2. 速率（20 秒采样，含暂存） ==="
tot1=$(du -sb "$PARTS" 2>/dev/null | awk '{print $1}')
sleep 20
tot2=$(du -sb "$PARTS" 2>/dev/null | awk '{print $1}')
echo "  合计 $(( tot2 / 1048576 )) MB / $(( EXP / 1048576 )) MB  ( $(awk -v a="$tot2" -v b="$EXP" 'BEGIN{printf "%.2f", 100*a/b}') % )"
awk -v a="$tot1" -v b="$tot2" 'BEGIN{printf "  速率 %.2f MB/s\n", (b-a)/20/1048576}'
awk -v a="$tot1" -v b="$tot2" -v e="$EXP" 'BEGIN{
  r=(b-a)/20; if (r>0) printf "  剩余约 %.0f 分钟\n", (e-b)/r/60; else print "  剩余：速率 0，无法估计"}'

echo; echo "=== 3. worker 进程 ==="
pgrep -af "curl .*model.safetensors" | head -8 | cut -c1-110 | sed 's/^/  /'
echo "  curl 进程数 $(pgrep -cf 'curl .*model.safetensors' || true)"

echo; echo "=== 4. 日志尾 ==="
tail -8 "$LOG" 2>/dev/null | sed 's/^/  /'

echo; echo "=== 5. API 返回结构（查 sha256 字段位置） ==="
if [ -s /tmp/_vlx_api.json ]; then
  echo "  字节 $(stat -c%s /tmp/_vlx_api.json)"
  python3 - <<'PY' 2>&1 | sed 's/^/  /'
import json
try:
    d = json.load(open("/tmp/_vlx_api.json", encoding="utf-8"))
except Exception as e:
    print("  解析失败:", type(e).__name__, str(e)[:120]); raise SystemExit
print("  顶层键:", list(d)[:20])
sib = d.get("siblings") or []
print("  siblings 条数:", len(sib))
for s in sib:
    if s.get("rfilename") == "model.safetensors":
        print("  条目键:", list(s))
        print("  内容:", json.dumps(s, ensure_ascii=False)[:400])
PY
else
  echo "  无 /tmp/_vlx_api.json"
fi
echo "=== 结束 ==="
