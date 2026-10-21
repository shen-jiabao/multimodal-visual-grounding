#!/bin/bash
# 诊断 Qwen3.8-27B 起服失败：Engine core initialization failed / shm_broadcast
set -u
R=/root/rivermind-data
LOG=$R/vllm38.log

echo "=== ① 共享内存 ==="
df -h /dev/shm | sed 's/^/  /'
echo "  /dev/shm 挂载项：$(grep -E ' /dev/shm ' /proc/mounts | cut -d' ' -f1-4)"
echo "  当前占用：$(ls -A /dev/shm 2>/dev/null | wc -l) 个文件"

echo; echo "=== ② 进程 ==="
echo "  启动器 $(pgrep -cf _q38_serve_run.sh || echo 0)｜vllm $(pgrep -cf 'vllm serve' || echo 0)"

echo; echo "=== ③ shm_broadcast 段（前后 12 行）==="
grep -n "shm_broadcast" "$LOG" | head -5 | sed 's/^/  /'
ln=$(grep -n "No available shared memory" "$LOG" | head -1 | cut -d: -f1)
if [ -n "${ln:-}" ]; then
  sed -n "$(( ln > 12 ? ln - 12 : 1 )),$(( ln + 12 ))p" "$LOG" | cut -c1-190 | sed 's/^/  /'
fi

echo; echo "=== ④ 逐行完整报错（ERROR 起 60 行）==="
first=$(grep -n "ERROR" "$LOG" | head -1 | cut -d: -f1)
if [ -n "${first:-}" ]; then
  sed -n "${first},$(( first + 60 ))p" "$LOG" | cut -c1-190 | sed 's/^/  /'
else
  echo "  （无 ERROR 行）"
fi

echo; echo "=== ⑤ 末尾 25 行 ==="
tail -25 "$LOG" | cut -c1-190 | sed 's/^/  /'
echo "=== 结束 ==="
