#!/bin/bash
# 等待 G 路线全量跑完，然后打印结算信息（供后台等待器调用）
set -u
R=/root/rivermind-data
OUT=$R/output/qwen32bf16_direct.json
LOG=$R/qwen32bf16_run.log

echo "=== 等待全量结束（最多 100 分钟） ==="
for i in $(seq 1 200); do
  if pgrep -f _qwen_full_pipeline.py >/dev/null; then
    [ $(( i % 10 )) -eq 0 ] && echo "  …已等 $(( i )) 分钟｜$(tail -1 "$LOG" 2>/dev/null | cut -c1-100)"
    sleep 60
  else
    echo "  ✓ 管线已结束（等待 $i 分钟）"
    break
  fi
done
pgrep -f _qwen_full_pipeline.py >/dev/null && { echo "  ★超时仍在跑"; exit 1; }

echo; echo "=== 日志尾 8 行 ==="
tail -8 "$LOG" 2>/dev/null | cut -c1-150 | sed 's/^/  /'

echo; echo "=== 结算 ==="
$R/venv/bin/python - "$OUT" "$R/_ordinal_cores_r2.json" <<'PY'
import json, sys, os
out, cores = sys.argv[1], sys.argv[2]
d = json.load(open(out, encoding="utf-8"))
co = set(json.load(open(cores, encoding="utf-8")))
non = {k: v for k, v in d.items() if k not in co}
print(f"  产物总条目 {len(d)}（其中非序数 {len(non)}）")
err = [k for k, v in non.items() if v.get("error")]
empty = [k for k, v in non.items() if not v.get("bbox")]
print(f"  err {len(err)}｜无框 {len(empty)}（{len(empty)/max(1,len(non)):.1%}）")
full = 4587
print(f"  ★非序数覆盖 {len(non)}/{full}"
      + ("  ✓ 完整" if len(non) >= full else f"  ✗ 缺 {full-len(non)}"))
tmp = [f for f in os.listdir(os.path.dirname(out)) if f.endswith(".tmp")]
print(f"  .tmp 残留 {len(tmp)}（应为 0）")
PY

echo; echo "=== 服务与 GPU ==="
echo "  vllm: $(pgrep -cf 'vllm serve' || true) 个（装配不需要 GPU，可保留）"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束（可装配） ==="