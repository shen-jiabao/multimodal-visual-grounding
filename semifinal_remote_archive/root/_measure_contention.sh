#!/bin/bash
# 量化并发摊薄：对比 LocateAnything 起跑前后，8B 主任务的图吞吐。
# 并在不干扰主线的前提下，看探针进程是否真的在推进（而非卡死）。
PY=/root/miniconda3/envs/myconda/bin/python
echo "===== A) 主任务吞吐分段（图/小时）====="
$PY - <<'PY'
import re
rows = []
for m in re.finditer(r"^\[(\d+)/1295\] (\S+) direct=(\d+) som=(\d+) 已用([\d.]+)h",
                     open("/srv/output/qwen_full.log", encoding="utf-8", errors="replace").read(), re.M):
    rows.append((int(m.group(1)), int(m.group(3)), float(m.group(5))))
print(f"  采样点 {len(rows)}，最新 [{rows[-1][0]}] 已用 {rows[-1][2]:.2f}h")
# 分段：LocateAnything 大约在 15:25 之后起跑；用 160 图作为分界近似
cut = 160
for lo, hi, tag in ((0, cut, "并发前"), (cut, rows[-1][0], "并发后")):
    seg = [(i, e) for i, d, e in rows if lo < i <= hi]
    if len(seg) < 2: continue
    di = seg[-1][0] - seg[0][0]; dt = seg[-1][1] - seg[0][1]
    print(f"  {tag} [{lo}~{hi}]: {di} 图 / {dt*60:.1f} min = {di/dt:.1f} 图/小时")
PY

echo
echo "===== B) 探针进程是否推进 ====="
ps -eo pid,etime,pcpu,pmem,stat,cmd | grep -E "_probe_locale|_probe_locate_semi" | grep -v grep
echo "--- 探针 GPU 占用 ---"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- 输出文件（探针只在结束时写）---"
ls -l /srv/probe_locate/out/ 2>/dev/null || echo "  尚无输出"

echo
echo "===== C) GPU 实时利用率采样（5 次，间隔 3s）====="
for i in 1 2 3 4 5; do
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  sleep 3
done
