#!/bin/bash
# 等 235B-AWQ 下载结束并核验权重完整性（不需要 GPU）
set -u
R=/root/rivermind-data
DIR=$R/models/Qwen3-VL-235B-A22B-Instruct-AWQ
LOG=$R/dl235b.log
V=$R/venv/bin/python

for i in $(seq 1 720); do
  pgrep -f 'snapshot_download' >/dev/null || break
  sleep 30
done
echo "下载进程已退出（等待约 $((i*30)) 秒）"
echo; echo "=== 日志尾 6 行 ==="
tail -6 "$LOG" 2>/dev/null
echo; echo "=== 目录体积 ==="
du -sh "$DIR" 2>/dev/null | sed 's/^/  /'
echo; echo "=== 权重完整性（分片逐一对索引核验）==="
"$V" - "$DIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("  ★无 index.json ⇒ 未下完"); raise SystemExit(9)
m = json.load(open(idx))["weight_map"]
sh = sorted(set(m.values()))
miss, short = [], []
tot = 0
for s in sh:
    p = os.path.join(d, s)
    if not os.path.exists(p):
        miss.append(s); continue
    sz = os.path.getsize(p); tot += sz
    if sz < 1024 * 1024:
        short.append((s, sz))
print(f"  索引要求分片 {len(sh)} 个｜缺失 {len(miss)}｜过小 {len(short)}")
print(f"  分片合计 {tot/1024**3:.2f} GB")
if miss:  print("  ★缺失:", miss[:5])
if short: print("  ★过小:", short[:5])
need = ["config.json", "preprocessor_config.json", "tokenizer_config.json",
        "generation_config.json"]
bad = [f for f in need if not os.path.exists(os.path.join(d, f))]
print("  关键配置缺失:", bad or "无")
print("  ⇒", "✓ 完整，可起服务" if not (miss or short or bad)
      else "★不完整，需续传（重跑 _dl_235b.sh 即可断点续传）")
PYEOF
echo; echo "=== 磁盘 ==="
df -h "$R" | tail -1 | sed 's/^/  /'
echo "=== 结束 ==="