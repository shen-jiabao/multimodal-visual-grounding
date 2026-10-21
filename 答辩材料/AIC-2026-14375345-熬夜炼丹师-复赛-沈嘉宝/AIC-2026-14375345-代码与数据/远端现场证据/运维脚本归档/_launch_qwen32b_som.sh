#!/bin/bash
# Phase B（序数 SoM）启动器：用 32B 跑**全部 1103 条序数**的 SoM 枚举+选号。
#
# 为什么现在跑：算力只在服务器存活期内可得。本步只**预取证据**，不启用判据——
# 是否采用仍由 `_frozen_criteria.md` 的触发条件（32B 非序数件读数 ≥ 0.770）决定。
# 跑全量 1103 而非冻结稿里的 407，是因为成本同量级（约 20 分钟）而覆盖面更宽：
# 若作用范围的口径日后需要复核，数据已在手；不影响"只用 407"的判据。
#
# 机制与线上一致（同族升级 8B SoM → 32B SoM）：按 (图, core) 分组，
# 每图每 core 枚举一次 → 叠加红编号 → 每条问句选号一次。
set -u
PY=${QWEN_PY:-/root/rivermind-data/venv/bin/python}
DATA=${QWEN_DATA:-/root/rivermind-data/dataset}
OUT=${QWEN_OUT_DIRECT:-/root/rivermind-data/output/qwen32b_direct.json}
SOMOUT=${QWEN_OUT_SOM:-/root/rivermind-data/output/qwen32b_som.json}
LOG=${QWEN_LOG:-/root/rivermind-data/qwen32b_som_run.log}
PIPE=${QWEN_PIPE:-/root/_qwen_full_pipeline.py}
CORES=${QWEN_CORES:-/root/rivermind-data/_ordinal_cores_v3.json}
API=${QWEN_API_BASE:-http://127.0.0.1:8000/v1}
fail=0

echo "===== 闸门 1) vLLM 在服 qwen3-vl-32b ====="
models=$(curl -s --max-time 10 "$API/models" || echo "")
echo "$models" | head -c 200; echo
echo "$models" | grep -q "qwen3-vl-32b" || { echo "FAIL: 未提供 qwen3-vl-32b"; fail=1; }

echo "===== 闸门 2) 数据与核心词表 ====="
nq=$("$PY" -c "import json;print(len(json.load(open('$DATA/queries/queries.json'))))" 2>/dev/null || echo 0)
nimg=$(ls "$DATA/Images/visible" 2>/dev/null | wc -l)
"$PY" -c "import json;print('  cores',len(json.load(open('$CORES'))))" 2>/dev/null || echo "  缺 cores"
echo "  queries=$nq  visible 图=$nimg"
[ "$nq" -ge 5690 ] && [ "$nimg" -ge 1295 ] || { echo "FAIL: 数据不齐"; fail=1; }

echo "===== 闸门 3) SoM 三常量完好 ====="
for pat in "ENUM_PROMPT" "SEL_PROMPT" "draw_som" "RE_NUM"; do
  n=$(grep -c "$pat" "$PIPE")
  echo "  $pat × $n"
  [ "$n" -ge 1 ] || { echo "FAIL: 缺 $pat"; fail=1; }
done
grep -n 'Detect ALL instances of' "$PIPE" | head -1

echo "===== 闸门 4) 输出可写、direct 产物未被动 ====="
touch "$SOMOUT" || { echo "FAIL: som 输出不可写"; fail=1; }
echo "  som 输出: $(ls -l "$SOMOUT" 2>/dev/null)"
echo "  direct 产物: $(ls -l "$OUT" 2>/dev/null)"
"$PY" -c "
import json
d = json.load(open('$OUT'))
print('  direct 条目', len(d))" 2>/dev/null || echo "  （direct 读取失败，不影响 Phase B）"

echo "===== 结论 ====="
if [ "$fail" -ne 0 ]; then echo "BLOCKED: 拒绝启动"; exit 2; fi
echo "PASS: 四道闸门全过"

if pgrep -f "_qwen_full_pipeline.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"; exit 0
fi

cd /root
QWEN_ENGINE=api32b \
QWEN_API_BASE="$API" \
QWEN_CONC=${QWEN_CONC:-8} \
QWEN_PHASE=B \
QWEN_ONLY_NONORD=0 \
QWEN_DATA="$DATA" \
QWEN_CORES="$CORES" \
QWEN_OUT_DIRECT="$OUT" \
QWEN_OUT_SOM="$SOMOUT" \
setsid nohup "$PY" -u "$PIPE" > "$LOG" 2>&1 < /dev/null &
sleep 25
echo "已启动 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"
echo "--- 日志首 20 行 ---"
head -n 20 "$LOG"