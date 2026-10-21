#!/bin/bash
# 等 8B 主任务结束后**自动**调用 LA 启动器，免去人工盯梢。
# 最终把关仍是 _launch_locate_semi.sh 的闸门 4（GPU 必须空闲）——
# 本脚本只负责"什么时候去试"，不负责"是否放行"。
#
# 三处加固（对应交接清单 #2/#3/#4）：
#   #3 进程退出 ≠ 产物完整 → 衔接前自动统计两份产物的条目数与 [ERR] 数
#   #4 启动后自动核对 LA 进程是否存活、显存是否被 LA 占用
#   #2 每 10 分钟写一行心跳（第 N 次检查），供人工判断监视器是否还活着
LOG=/srv/output/watch_launch.log
PY=/root/miniconda3/envs/myconda/bin/python
exec > >(tee -a $LOG) 2>&1
echo "=== [$(date)] 监视 8B 主任务，结束即启动 LA ==="

# 模式串是 _qwen_full_pipeline.py；本脚本 cmdline 为
# `bash /root/_watch_then_launch.sh`，不含该串，故不会自匹配。
n=0
while pgrep -f '_qwen_full_pipeline.py' > /dev/null; do
  n=$((n + 1))
  if [ $((n % 10)) -eq 1 ]; then
    printf '[%s] 心跳 第 %d 次检查：8B 仍在运行\n' "$(date +%H:%M:%S)" "$n"
  fi
  sleep 60
done

echo "[$(date)] 8B 主任务进程已退出；等 45s 让显存释放..."
sleep 45
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo
echo "=== [#3] 8B 产物完整性核对（进程退出不代表产物完整）==="
$PY - <<'PY'
import json, os
for p in ("/srv/output/qwen3vl_all.json", "/srv/output/qwen3vl_som_all.json"):
    if not os.path.exists(p):
        print(f"  {p}: 缺失")
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception as e:
        print(f"  {p}: 解析失败 {e!r}")
        continue
    nb = sum(1 for v in d.values() if v.get("bbox"))
    ne = sum(1 for v in d.values() if v.get("err"))
    print(f"  {os.path.basename(p)}: {len(d)} 条 | 有框 {nb} | err {ne}")
print("  期望: direct 5690 条、som 1103 条")
PY
echo -n "  qwen_full.log 中 [ERR] 计数: "
grep -c '\[ERR\]' /srv/output/qwen_full.log

echo
echo "[$(date)] 调用 LA 启动器（其闸门 4 做最终判断）..."
bash /root/_launch_locate_semi.sh
echo "[$(date)] 启动器返回码 rc=$?"

echo
echo "=== [#4] 启动后核对 ==="
sleep 20
echo "--- LA 进程 ---"
ps -eo pid,etime,cmd | grep -E '_locate_full\.py' | grep -v grep
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "--- LA 日志前 12 行 ---"
head -n 12 /srv/output/locate_full_stdout.log 2>/dev/null || echo "  (日志尚未生成)"
echo "=== [$(date)] 衔接流程结束 ==="
