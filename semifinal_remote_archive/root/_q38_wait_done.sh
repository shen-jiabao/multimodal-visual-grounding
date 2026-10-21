#!/bin/bash
# 等 Qwen3.8-27B 下载完成 → 独立复核（G1）→ 打印结算
#   下载与合并由远端 monitor/worker 分离执行，本脚本只等 + 验。
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B
MODEL=Qwen/Qwen3.8-27B

echo "=== 0) 当前状态 ==="
echo "  worker 数：$(pgrep -cf '_q38_w.sh' || true)｜monitor：$(pgrep -cf '_q38_monitor.sh' || true)"
echo "  目录：$(du -sh "$D" 2>/dev/null | cut -f1)"
echo "  可用磁盘：$(df -BG --output=avail $R | tail -1 | tr -dc '0-9') GB"
echo "  AWQ 是否已删：$([ -d $R/models/Qwen3-VL-32B-Instruct-AWQ ] && echo '仍存在 ★' || echo '已删除 ✓')"

echo; echo "=== 1) 等待下载与合并完成（最多 110 分钟） ==="
prev=0
for i in $(seq 1 110); do
  mon=$(pgrep -cf '_q38_monitor.sh' || true)
  gb=$(du -sm "$D" 2>/dev/null | cut -f1)
  if [ "${mon:-0}" -eq 0 ]; then
    echo "  ✓ monitor 已退出（等待 $i 分钟）"
    break
  fi
  # 停滞检测：20 分钟无增长则告警（不退出，继续等）
  if [ $(( i % 20 )) -eq 0 ]; then
    if [ "${gb:-0}" -le "${prev:-0}" ]; then
      echo "  ★告警：20 分钟无增长（${gb} MB）⇒ 可能停滞"
      echo "    下载日志尾："; tail -6 "$R/q38_dl.log" 2>/dev/null | cut -c1-140 | sed 's/^/      /'
    fi
    prev=${gb:-0}
  fi
  [ $(( i % 5 )) -eq 0 ] && echo "    …${i} 分钟｜$(( ${gb:-0} / 1024 )) GB｜worker $(pgrep -cf '_q38_w.sh' || true)"
  sleep 60
done

echo; echo "=== 2) 下载日志尾 12 行 ==="
tail -12 "$R/q38_dl.log" 2>/dev/null | cut -c1-150 | sed 's/^/  /'

echo; echo "=== 3) ★G1：逐片 sha256 独立复核 ==="
curl -sL --max-time 90 "https://hf-mirror.com/api/models/$MODEL?blobs=true" -o /tmp/_q38v.json 2>/dev/null
python3 - <<'PY' > /tmp/_q38_manifest.tsv 2>/dev/null
import json
try:
    d = json.load(open("/tmp/_q38v.json", encoding="utf-8"))
except Exception:
    raise SystemExit(0)
for s in d.get("siblings") or []:
    n = s["rfilename"]
    if not n.endswith(".safetensors"): continue
    sz = s.get("size") or (s.get("lfs") or {}).get("size") or 0
    sha = (s.get("lfs") or {}).get("sha256") or ""
    print(f"{n}\t{sz}\t{sha}")
PY
echo "  manifest 分片 $(grep -c . /tmp/_q38_manifest.tsv 2>/dev/null || echo 0) 个"

ok=0; bad=0
while IFS=$'\t' read -r name size sha; do
  [ -z "$name" ] && continue
  f="$D/$name"
  if [ ! -f "$f" ]; then echo "  ✗ 缺失 $name"; bad=$((bad+1)); continue; fi
  s=$(stat -c%s "$f")
  if [ "$s" != "$size" ]; then echo "  ✗ 大小不符 $name ($s≠$size)"; bad=$((bad+1)); continue; fi
  g=$(sha256sum "$f" | awk '{print $1}')
  if [ "$g" = "$sha" ]; then
    ok=$((ok+1))
    [ $(( ok % 5 )) -eq 0 ] && echo "  …已验 $ok 片"
  else
    echo "  ✗ sha256 不符 $name"; echo "      得 $g"; echo "      期 $sha"; bad=$((bad+1))
  fi
done < /tmp/_q38_manifest.tsv
echo "  ⇒ **通过 $ok ｜失败 $bad**"

echo; echo "=== 4) index.json 覆盖性 ==="
$R/venv/bin/python - "$D" <<'PY'
import json, os, glob, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(idx):
    print("  ★无 index.json"); raise SystemExit(0)
m = json.load(open(idx, encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
on = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "model-*.safetensors")))
print(f"  权重条目 {len(m)}｜引用分片 {len(sh)}｜盘上 {len(on)}")
print(f"  缺失 {[s for s in sh if s not in on][:3]}｜多余 {[s for s in on if s not in sh][:3]}")
PY

echo; echo "=== 5) 结算 ==="
du -sh "$D" 2>/dev/null | sed 's/^/  /'
df -h $R | tail -1 | sed 's/^/  /'
echo "  AWQ：$([ -d $R/models/Qwen3-VL-32B-Instruct-AWQ ] && echo '仍存在' || echo '已删除 ✓')"
echo "  GPU：$(ls /dev/nvidia[0-9]* >/dev/null 2>&1 && nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tr '\n' ' ' || echo '✗ 尚未分配')"
echo "=== 结束（下载完成；起服前需 GPU） ==="