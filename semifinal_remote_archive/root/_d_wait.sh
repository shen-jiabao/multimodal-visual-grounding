#!/bin/bash
# 等小样跑完，然后清点"错误类型"——err 会污染改动率，必须先看清是什么错
set -u
R=/root/rivermind-data
for i in $(seq 1 120); do
  pgrep -f 'qwen_full_pipeline' >/dev/null || break
  sleep 10
done
echo "管线已退出（等待约 $((i*10)) 秒）"
echo; echo "=== 日志尾部 ==="
tail -4 "$R/d_sample_run.log"
echo; echo "=== 产物清点 + 错误分类 ==="
"$R/venv/bin/python" - "$R/output/qwen32b_d3_sample.json" <<'PYEOF'
import json
from collections import Counter
p = "/root/rivermind-data/output/qwen32b_d3_sample.json"
d = json.load(open(p, encoding="utf-8"))
nb = sum(1 for v in d.values() if v.get("bbox"))
errs = Counter(v["err"][:80] for v in d.values() if v.get("err"))
print(f"  条数 {len(d)}｜有框 {nb}｜输出[]无框 {len(d)-nb-sum(errs.values())}"
      f"｜err {sum(errs.values())}")
for k, n in errs.most_common(8):
    print(f"    err × {n}: {k}")
# 有框条目的 raw 是否正常（是否出现格式漂移）
bad = [k for k, v in d.items()
       if v.get("bbox") and "bbox_2d" not in (v.get("raw") or "")]
print(f"  有框但 raw 缺 bbox_2d 的: {len(bad)}")
PYEOF
echo; echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== 结束 ==="