#!/bin/bash
# G 路线（bf16 32B 全量）进度与健康度
# 盯三件：err 计数、空框率、以及产物原子性（.tmp 不应残留）
set -u
R=/root/rivermind-data
OUT=$R/output/qwen32bf16_direct.json
LOG=$R/qwen32bf16_run.log

echo "=== 进程 ==="
pid=$(pgrep -f _qwen_full_pipeline.py | head -1)
echo "  管线 PID=${pid:-无}"
echo "  vllm : $(pgrep -cf 'vllm serve' || true) 个"

echo; echo "=== 进度（日志尾 5 行） ==="
tail -5 "$LOG" 2>/dev/null | cut -c1-150 | sed 's/^/  /'

echo; echo "=== 产物健康度 ==="
$R/venv/bin/python - "$OUT" "$LOG" <<'PY'
import json, os, re, sys
out, log = sys.argv[1], sys.argv[2]
try:
    d = json.load(open(out, encoding="utf-8"))
except Exception as e:
    print(f"  ★产物读取失败：{type(e).__name__}: {e}"); raise SystemExit(0)
print(f"  已落盘条目 {len(d)}")
err = [k for k, v in d.items() if isinstance(v, dict) and v.get("error")]
print(f"  err 条目 {len(err)}"
      + (f"  ← 样例 {err[:3]}" if err else ""))
empty = [k for k, v in d.items()
         if isinstance(v, dict) and not v.get("bbox")]
print(f"  无框条目 {len(empty)}"
      + (f"  ({len(empty)/max(1,len(d)):.1%})" if d else ""))
ok = [v for v in d.values() if isinstance(v, dict) and v.get("bbox")]
if ok:
    areas = []
    for v in ok:
        b = v["bbox"]
        areas.append((b[2]-b[0])*(b[3]-b[1]))
    areas.sort()
    print(f"  框面积 p50={areas[len(areas)//2]:.4f}  "
          f"p95={areas[int(len(areas)*0.95)]:.4f}  max={areas[-1]:.4f}")
    print(f"  面积>0.35 的 {sum(1 for a in areas if a > 0.35)} 条（装配时会保留基线）")
# 残片检查
tmp = [f for f in os.listdir(os.path.dirname(out)) if f.endswith(".tmp")]
print(f"  目录内 .tmp 残留 {len(tmp)} 个（原子落盘应为 0）")
PY

echo; echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed 's/^/  /'
echo "=== 结束 ==="