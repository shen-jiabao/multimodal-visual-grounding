#!/bin/bash
# 停掉 LA 全量，改为只用 LA 对 45 条无框跑红外补轮。
# 顺序很关键：先停补轮监视器（否则它会检测到 LA 退出而自动触发，且被自己的
# "LA 产物完整性"闸门拦住，产生噪音），再停 LA，最后手动起补轮。
set -u
VPY=/srv/venv-la2/bin/python
echo "=== [$(date)] 切换到「只跑 45 条红外补轮」 ==="

echo
echo "== 1) 停掉补轮监视器（防竞态）=="
W=$(pgrep -f '_watch_then_retry_ir\.sh')
echo "  监视器 PID: ${W:-无}"
[ -n "$W" ] && kill $W 2>/dev/null
sleep 2
echo -n "  停止后残留: "; pgrep -f '_watch_then_retry_ir\.sh' | wc -l

echo
echo "== 2) 停掉 LA 全量 =="
L=$(pgrep -f '_locate_full\.py')
echo "  LA PID: ${L:-无}"
[ -n "$L" ] && kill $L 2>/dev/null
for i in $(seq 1 15); do
  sleep 2
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$n" = "0" ]; then echo "  显存已释放（第 ${i} 次检查）"; break; fi
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo
echo "== 3) 处理 LA 全量的半成品（避免被误当成完整产物）=="
if [ -f /srv/output/locate_bbox_all.json ]; then
  $VPY -c "import json;d=json.load(open('/srv/output/locate_bbox_all.json'));print('  半成品条目数:',len(d))"
  mv /srv/output/locate_bbox_all.json /srv/output/locate_bbox_all.partial_visible.json
  echo "  已改名为 locate_bbox_all.partial_visible.json（可见光口径、不完整，不参与任何合并）"
else
  echo "  无半成品（首检查点 50 条尚未落盘）"
fi

echo
echo "== 4) 起 45 条红外补轮 =="
cd /root
setsid nohup $VPY -u /root/_retry_ir.py > /srv/output/retry_ir_stdout.log 2>&1 < /dev/null &
sleep 15
echo "  PID: $(pgrep -f '_retry_ir\.py' | tr '\n' ' ')"
echo "--- 日志 ---"
cat /srv/output/retry_ir_stdout.log 2>/dev/null | tail -n 12
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
