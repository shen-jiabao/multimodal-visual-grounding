#!/bin/bash
# 权重终检：以 index.json 为准核 18 分片是否齐全，并统计 sha256 双验通过/失败条数
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B

echo "=== ① index.json 口径（权威） ==="
$R/venv_q38/bin/python - "$D" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("  ★无 index.json"); sys.exit(9)
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
miss = [s for s in sh if not os.path.exists(os.path.join(d, s))]
tot = sum(os.path.getsize(os.path.join(d, s)) for s in sh if os.path.exists(os.path.join(d, s)))
print(f"  分片 {len(sh)} 个｜缺失 {len(miss)} 个｜合计 {tot/1024**3:.2f} GiB｜权重条目 {len(m)}")
if miss:
    print("  ★缺失:", miss)
    sys.exit(9)
print("  ✓ 18 分片齐全")
PY
rc=$?
[ $rc -ne 0 ] && echo "  ⇒ 权重仍未就绪"

echo
echo "=== ② 下载日志统计 ==="
L=$R/q38_dl.log
if [ -f "$L" ]; then
  ok=$(grep -ac "双验通过" "$L")
  ng=$(grep -ac "★" "$L")
  echo "  sha256+大小 双验通过条数: $ok"
  echo "  异常（★）条数: $ng"
  [ "$ng" -gt 0 ] && grep -a "★" "$L" | tail -5 | sed 's/^/    /'
  echo "  --- 末尾 5 行 ---"
  tail -5 "$L" | sed 's/^/    /'
fi

echo
echo "=== ③ 进程状态 ==="
echo "  monitor: $(pgrep -cf _q38_monitor.sh 2>/dev/null || echo 0)"
echo "  worker:  $(pgrep -cf _q38_w.sh 2>/dev/null || echo 0)"
echo "  curl:    $(pgrep -cf 'Qwen3.8-27B/resolve' 2>/dev/null || echo 0)"

echo
echo "=== ④ 目录与磁盘 ==="
echo "  $D = $(( $(du -sb "$D" 2>/dev/null | awk '{print $1}') / 1048576 )) MB"
ls -1 "$D" | wc -l | sed 's/^/  文件数: /'
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== DONE ==="
