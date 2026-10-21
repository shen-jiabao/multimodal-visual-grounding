#!/bin/bash
# 32B 全量（仅非序数直接定位）启动器：六道闸门全过 + 烟测通过，才放行。
#
# 目的：用 Qwen3-VL-32B-AWQ 重跑 4587 条**非序数**直接定位，替换现提交件里那一桶。
# 依据（归零法实测）：非序数准确率仅 79.5%（940 错，占全部错题 69%），是到 0.8 的
# 唯一主攻方向；序数层保持不动以隔离变量（该件得分本身即测量：
# 得分×5690 − 675 = 32B 的非序数正确数）。
#
# 路径全部可经环境变量覆盖（缺省为 gpuhome 新机的布局）；同一份实现两处可用。
set -u
PY=${QWEN_PY:-/root/rivermind-data/venv/bin/python}
DATA=${QWEN_DATA:-/root/rivermind-data/dataset}
OUT=${QWEN_OUT:-/root/rivermind-data/output/qwen32b_direct.json}
LOG=${QWEN_LOG:-/root/rivermind-data/qwen32b_run.log}
PIPE=${QWEN_PIPE:-/root/_qwen_full_pipeline.py}
CORES=${QWEN_CORES:-/root/rivermind-data/_ordinal_cores_v3.json}
API=${QWEN_API_BASE:-http://127.0.0.1:8000/v1}
SOMOUT=${QWEN_OUT_SOM:-/root/rivermind-data/output/qwen32b_som.json}
CONC=${QWEN_CONC:-8}
fail=0

echo "===== 闸门 1) vLLM 是否在服 qwen3-vl-32b ====="
models=$(curl -s --max-time 10 "$API/models" || echo "")
echo "$models" | head -c 300; echo
if ! echo "$models" | grep -q "qwen3-vl-32b"; then
  echo "FAIL: $API 未提供 qwen3-vl-32b"; fail=1
fi

echo "===== 闸门 2) 数据齐备 ====="
nq=$($PY -c "import json;print(len(json.load(open('$DATA/queries/queries.json'))))" 2>/dev/null || echo 0)
nimg=$(ls "$DATA/Images/visible" 2>/dev/null | wc -l)
echo "queries=$nq  visible 图=$nimg"
if [ "$nq" -lt 5690 ] || [ "$nimg" -lt 1295 ]; then echo "FAIL: 数据不齐（需 5690 / 1295）"; fail=1; fi

echo "===== 闸门 3) 序数核心词表 ====="
if [ -f "$CORES" ]; then
  $PY -c "import json;d=json.load(open('$CORES'));print('  cores',len(d))"
else
  echo "FAIL: 缺 $CORES（无法切出非序数集）"; fail=1
fi

echo "===== 闸门 4) 输出与磁盘 ====="
mkdir -p "$(dirname "$OUT")"
touch "$OUT" || { echo "FAIL: 输出不可写"; fail=1; }
df -h "$(dirname "$OUT")" | tail -1

echo "===== 闸门 5) 管线常量完好 ====="
for pat in "FULL_DIRECT_PROMPT" "ENUM_PROMPT" "SEL_PROMPT" "QWEN_ENGINE" "make_gen_api"; do
  n=$(grep -c "$pat" "$PIPE")
  echo "  $pat × $n"
  [ "$n" -ge 1 ] || { echo "FAIL: 缺 $pat"; fail=1; }
done
grep -n "^ENGINE = " "$PIPE"
grep -n "Locate the object described below" "$PIPE" | head -1

echo "===== 闸门 6) 烟测（接线自校验，必须逐位一致）====="
QWEN_ENGINE=api32b QWEN_API_BASE="$API" QWEN_DATA="$DATA" \
  $PY -u /root/_smoke_qwen32b.py --data "$DATA" 2>&1 | tail -30
smoke=${PIPESTATUS[0]}
echo "烟测退出码 $smoke"
[ "$smoke" -eq 0 ] || { echo "FAIL: 接线自校验未通过（见上）"; fail=1; }

echo "===== 结论 ====="
if [ "$fail" -ne 0 ]; then echo "BLOCKED: 拒绝启动全量"; exit 2; fi
echo "PASS: 六道闸门全过"

if pgrep -f "_qwen_full_pipeline.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"; exit 0
fi

cd /root
QWEN_ENGINE=api32b \
QWEN_API_BASE="$API" \
QWEN_CONC=$CONC \
QWEN_PHASE=A \
QWEN_ONLY_NONORD=1 \
QWEN_DATA="$DATA" \
QWEN_CORES="$CORES" \
QWEN_OUT_DIRECT="$OUT" \
QWEN_OUT_SOM="$SOMOUT" \
setsid nohup $PY -u "$PIPE" > "$LOG" 2>&1 < /dev/null &
sleep 30
echo "已启动 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"
echo "--- 日志首 20 行 ---"
head -n 20 "$LOG"