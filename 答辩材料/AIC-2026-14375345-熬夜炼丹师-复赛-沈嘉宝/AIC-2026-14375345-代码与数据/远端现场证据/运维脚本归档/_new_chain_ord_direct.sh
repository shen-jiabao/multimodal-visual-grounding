#!/bin/bash
# 链式预取：等 Phase B（序数 SoM）跑完，再跑「32B 直接框 on 1103 序数」作保险。
#
# 为什么链式而不并发：Phase B 是单流请求、GPU 已近饱和（实测 96%），并发只会互相
# 拖慢；且 SoM 是冻结稿的关键路径，不能被保险项拖累。
# 为什么 setsid nohup 起在远端：本步可能在我方会话之外进行，不应依赖 SSH 存活。
#
# ★产物另存 qwen32b_direct_ord.json，**绝不覆盖**已验收的
#   qwen32b_direct.json（sha256 b8e2f9e24fb24594…）。
# 做法：先把源产物（4587 条非序数）克隆过去，于是 Phase A 只剩 1103 条序数待跑。
set -u
PY=/root/rivermind-data/venv/bin/python
PIPE=/root/_qwen_full_pipeline.py
DATA=/root/rivermind-data/dataset
CORES=/root/rivermind-data/_ordinal_cores_v3.json
SRC=/root/rivermind-data/output/qwen32b_direct.json
OUT=/root/rivermind-data/output/qwen32b_direct_ord.json
SOMDUMMY=/root/rivermind-data/output/_som_dummy.json
LOGQ=/root/rivermind-data/qwen32b_ord_direct.log

echo "[chain] 等待 Phase B 结束 $(date -Is)"
while pgrep -f "_qwen_full_pipeline" > /dev/null; do sleep 20; done
echo "[chain] Phase B 已退出 $(date -Is)"

n=$("$PY" -c "import json;print(len(json.load(open('$SRC'))))" 2>/dev/null || echo 0)
echo "[chain] 源 direct 条目 $n（应 4587）"
if [ "$n" -ne 4587 ]; then echo "[chain] 源产物异常，放弃保险项"; exit 3; fi
cp -f "$SRC" "$OUT"
echo "[chain] 已克隆 → $OUT（覆盖原 ord 产物，若有）"

cd /root
QWEN_ENGINE=api32b \
QWEN_API_BASE=http://127.0.0.1:8000/v1 \
QWEN_CONC=8 \
QWEN_PHASE=A \
QWEN_ONLY_NONORD=0 \
QWEN_DATA="$DATA" \
QWEN_CORES="$CORES" \
QWEN_OUT_DIRECT="$OUT" \
QWEN_OUT_SOM="$SOMDUMMY" \
setsid nohup "$PY" -u "$PIPE" > "$LOGQ" 2>&1 < /dev/null &
sleep 20
echo "[chain] 保险项已启动 PID=$(pgrep -f _qwen_full_pipeline | head -1)"
head -n 8 "$LOGQ"