#!/bin/bash
# Qwen3.8-27B 全量进度（可中途看，口径与装配器一致）
set -u
R=/root/rivermind-data
OUT=$R/output/qwen38_direct.json
LOG=$R/qwen38_run.log

echo "=== 进程 ==="
echo "  管线 $(pgrep -cf _qwen_full_pipeline.py || echo 0) 个｜vllm $(pgrep -cf 'vllm serve' || echo 0) 个"
echo "  服务：$(curl -s --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -o qwen38-27b | head -1 || echo '✗ 未响应')"

echo; echo "=== 日志尾 8 行 ==="
tail -8 "$LOG" 2>/dev/null | cut -c1-190 | sed 's/^/  /'

echo; echo "=== 产物健康度 ==="
$R/venv/bin/python - "$OUT" <<'PY' 2>&1 | sed 's/^/  /'
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p):
    print("产物尚未生成"); raise SystemExit(0)
try:
    d = json.load(open(p, encoding="utf-8"))
except Exception as e:
    print("解析失败（可能正在写入）:", type(e).__name__); raise SystemExit(0)
boxed = [k for k, v in d.items() if v.get("bbox")]
empty = [k for k, v in d.items() if not v.get("bbox")]
print(f"已出 {len(d)} 条｜有框 {len(boxed)}（{len(boxed)/max(len(d),1):.1%}）"
      f"｜输出 [] {len(empty)}（{len(empty)/max(len(d),1):.1%}）")
ars = sorted((v["bbox"][2]-v["bbox"][0]) * (v["bbox"][3]-v["bbox"][1])
             for v in d.values() if v.get("bbox"))
if ars:
    qs = lambda t: ars[min(int(len(ars)*t), len(ars)-1)]
    print(f"框面积 p10 {qs(.1):.4f} / p50 {qs(.5):.4f} / p90 {qs(.9):.4f} / max {ars[-1]:.4f}")
    print(f"面积>0.35 的 {sum(1 for a in ars if a > 0.35)} 条"
          f"｜<1e-4 的 {sum(1 for a in ars if a < 1e-4)} 条"
          f"｜整幅(≥0.99) 的 {sum(1 for a in ars if a >= 0.99)} 条")
PY

echo; echo "=== 异常行 ==="
grep -icE "error|traceback|err\"|失败" "$LOG" 2>/dev/null | sed 's/^/  计数 /'
grep -iE "error|traceback|失败" "$LOG" 2>/dev/null | tail -4 | cut -c1-170 | sed 's/^/  ✗ /'
echo "=== DONE ==="
