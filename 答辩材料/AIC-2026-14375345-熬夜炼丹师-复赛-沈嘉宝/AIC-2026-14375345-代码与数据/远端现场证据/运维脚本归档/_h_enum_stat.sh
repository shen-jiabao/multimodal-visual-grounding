#!/bin/bash
# H 枚举任务：状态 + 枚举率（纯 bash/脚本，避免 PowerShell 破坏内联命令）
set -u
R=/root/rivermind-data

echo "=== 进程 ==="
pgrep -af _qwen_full_pipeline.py | head -2 | cut -c1-110 | sed 's/^/  /' || echo "  （无：已结束）"

echo; echo "=== 枚举日志尾 8 行 ==="
tail -8 "$R/axis154_enum.log" 2>/dev/null | cut -c1-150 | sed 's/^/  /' || echo "  （无日志）"

echo; echo "=== 产物 ==="
for f in "$R/output/axis154_som.json" "$R/_axis154.json"; do
  if [ -f "$f" ]; then echo "  ✓ $f  $(stat -c%s "$f") 字节"; else echo "  ✗ 缺 $f"; fi
done

echo; echo "=== queries.json 是否有 answer 字段（核实用户给的口径是否成立） ==="
$R/venv/bin/python - "$R/dataset/queries/queries.json" <<'PY'
import json, sys
q = json.load(open(sys.argv[1], encoding="utf-8"))
it = list(q.items())[:1]
k, v = it[0]
print(f"  样例键：{sorted(v.keys())}")
print(f"  是否含 answer: {'answer' in v}")
print(f"  是否含任何 'gold/gt/bbox_gt' 类字段: "
      f"{[x for x in v if x.lower() in ('answer','gold','gt','bbox_gt','label')]}")
PY

echo; echo "=== 枚举率（若产物已生成） ==="
if [ -f "$R/output/axis154_som.json" ]; then
  $R/venv/bin/python - "$R/output/axis154_som.json" <<'PY'
import json, sys, collections
d = json.load(open(sys.argv[1], encoding="utf-8"))
n = collections.Counter(v.get("n_cands", 0) for v in d.values())
tot = len(d)
ok = sum(c for k, c in n.items() if k >= 2)
print(f"  产物 {tot} 条｜n_cands 分布 {dict(sorted(n.items()))}")
print(f"  ★全部 154 条枚举率（n_cands≥2）= {ok}/{tot} = {ok/max(1,tot):.1%}")
print(f"    参照：序数 SoM 枚举失败率 50.1% ⇒ 此处失败率 {100-100*ok/max(1,tot):.1f}%")
print(f"  reason 分布：{dict(collections.Counter(v.get('reason','ok') for v in d.values()))}")
PY
else
  echo "  （尚未生成）"
fi
echo "=== 结束 ==="