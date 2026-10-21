#!/bin/bash
# 诊断：缺失分片是"没开始"还是"下了一半"；并看 worker 分工与存活情况
set -u
R=/root/rivermind-data
D=$R/models/Qwen3.8-27B
P=$D/.parts

echo "=== ① 每个分片的块完成度 ==="
$R/venv_q38/bin/python - "$D" "$P" <<'PY'
import json, os, sys, glob
d, p = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(d, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
sh = sorted(set(m.values()))
CPF = 8
for name in sh:
    out = os.path.join(d, name)
    tot = os.path.getsize(os.path.join(d, name)) if False else None
    # 期望大小从 .parts 或已合并文件推断：用 index.json 不含大小 ⇒ 用 parts 之和 + 已存在文件
    if os.path.exists(out):
        print(f"  {name}  ✓已合并 {os.path.getsize(out)/1048576:.0f} MB")
        continue
    ps = sorted(glob.glob(os.path.join(p, name + ".part_*")))
    sm = {}
    for x in ps:
        if x.endswith(".tmp"): continue
        i = int(x.rsplit("_", 1)[1])
        sm[i] = os.path.getsize(x)
    got = sum(sm.values())
    print(f"  {name}  块 {len(sm)}/{CPF}  已下 {got/1048576:.0f} MB  明细 {[sm.get(i,0)//1048576 for i in range(CPF)]}")
PY

echo
echo "=== ② worker 分工与存活 ==="
for i in $(seq 0 15); do
  n=$(wc -l < "/tmp/_q38_wl_$i" 2>/dev/null || echo 0)
  printf "  wl_%d: %s 任务\n" "$i" "$n"
done
echo "  --- 存活进程 ---"
pgrep -af "_q38_w|_q38_monitor" 2>/dev/null | sed 's/^/    /' || echo "    （无）"

echo
echo "=== ③ 存活 worker 正在下什么（由 /proc 推断） ==="
for pid in $(pgrep -f "_q38_w.sh" 2>/dev/null); do
  echo "  pid $pid:"
  ls -l "/proc/$pid/fd" 2>/dev/null | grep -oE "\.parts/[^ ]+|_q38_wl_[0-9]+" | sort -u | sed 's/^/    /'
  ps -o etime= -p "$pid" 2>/dev/null | sed 's/^/    已运行 /'
done

echo
echo "=== ④ 下载日志末 25 行 ==="
tail -25 "$R/q38_dl.log" 2>/dev/null | sed 's/^/  /'
echo "=== DONE ==="
