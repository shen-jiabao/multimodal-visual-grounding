#!/bin/bash
# 卡数检查 + （≥2 张时）自动起服
#   卡未到位 → 报告并退出（不空转）
#   卡到位   → 后台分离执行 _get_serve_q38.sh，结果落日志
set -u
R=/root/rivermind-data
NG=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "当前 GPU 数：$NG"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo "  已跑 vllm：$(pgrep -cf 'vllm serve' || echo 0) 个"

if [ "${NG:-0}" -lt 2 ]; then
  echo "★仍不足 2 张 ⇒ 不起服"
  exit 0
fi

echo
echo "=== 已有 2 张 ⇒ 起服（后台分离）==="
pkill -f _q38_serve_run.sh 2>/dev/null || true
pkill -f "_get_serve_q38.sh" 2>/dev/null || true
sleep 2

cat > /tmp/_q38_serve_run.sh <<'EOS'
#!/bin/bash
set -u
R=/root/rivermind-data
{
  bash $R/_get_serve_q38.sh
} > $R/_q38_serve.log 2>&1
echo "=== SERVE-DONE ===" >> $R/_q38_serve.log
EOS
chmod +x /tmp/_q38_serve_run.sh
cp -f /tmp/_q38_serve_run.sh $R/_q38_serve_run.sh
: > "$R/_q38_serve.log"
cd $R
setsid nohup bash "$R/_q38_serve_run.sh" > /dev/null 2>&1 < /dev/null &
sleep 3
echo "已启动起服流程；进程 $(pgrep -cf _q38_serve_run.sh || echo 0) 个"
echo "日志：$R/_q38_serve.log"