#!/bin/bash
# M 件全量进度：条数 / 速率 / ETA / GPU 利用率（利用率高 = 已算力受限，并行无益）
set -u
R=/root/rivermind-data
OUT=$R/output/m_direct.json
LOG=$R/m_run.log
NQ=5690

echo "=== 进程 ==="
if pgrep -f "_qwen_full_pipeline.py" >/dev/null 2>&1; then
  echo "  运行中 pid $(pgrep -f _qwen_full_pipeline.py | tr '\n' ' ')"
else
  echo "  ★已结束"
fi
echo "  vllm: $(pgrep -cf 'vllm serve' || echo 0) 个"

echo
echo "=== 配置自证（首行）==="
grep -m1 "\[pipe\] 模态=" "$LOG" 2>/dev/null | sed 's/^/  /'

echo
echo "=== 进度 ==="
$R/venv_q38/bin/python - "$OUT" "$NQ" <<'PY'
import json, os, sys
p, nq = sys.argv[1], int(sys.argv[2])
d = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
ok = sum(1 for v in d.values() if v.get("bbox"))
print(f"  已出 {len(d)} / {nq} 条（{len(d)/nq*100:.1f}%）｜有框 {ok}"
      f"｜有框率 {ok/max(1,len(d))*100:.1f}%")
PY

echo
echo "=== 日志末 4 行 ==="
tail -4 "$LOG" 2>/dev/null | cut -c1-150 | sed 's/^/  /'

echo
echo "=== GPU（利用率高⇒算力受限；低⇒有并行空间）==="
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv,noheader 2>&1 | sed 's/^/  /'

echo
echo "=== 60 秒实测速率 ==="
cnt() {
  $R/venv_q38/bin/python -c "
import json, os
p = '$OUT'
print(len(json.load(open(p, encoding='utf-8'))) if os.path.exists(p) else 0)"
}
c1=$(cnt); sleep 60; c2=$(cnt)
awk -v a="$c1" -v b="$c2" -v n="$NQ" 'BEGIN{
  d=b-a; printf "  60 秒新增 %d 条 ⇒ %.1f 条/分\n", d, d;
  if (d>0) printf "  剩余 %d 条 ⇒ 约 %.2f 小时\n", n-b, (n-b)/d/60;
  else printf "  （本窗口无新增）\n";
}'
echo "=== DONE ==="
