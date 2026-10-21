#!/bin/bash
# G 路线全量启动器：bf16 Qwen3-VL-32B 重跑 4587 条**非序数**直接定位
#   与 _launch_qwen32b.sh（AWQ 那次）的唯一差别（单变量前提）：
#     · 服务模型名 → qwen3-vl-32b-bf16
#     · 输出/日志两个路径 → qwen32bf16_*
#     · 权重目录名断言（必须 32B-Instruct 且不含 AWQ）
#   提示词 / 解码 / 并发**逐字不变**。
#   六道闸门**一条不减**（与 AWQ 那次同口径）。
set -u
R=/root/rivermind-data
PY=${QWEN_PY:-$R/venv/bin/python}
DATA=${QWEN_DATA:-$R/dataset}
OUT=${QWEN_OUT:-$R/output/qwen32bf16_direct.json}
LOG=${QWEN_LOG:-$R/qwen32bf16_run.log}
SOMOUT=${QWEN_OUT_SOM:-$R/output/qwen32bf16_som.json}
API=${QWEN_API_BASE:-http://127.0.0.1:8000/v1}
API_MODEL=${QWEN_API_MODEL:-qwen3-vl-32b-bf16}
CONC=${QWEN_CONC:-8}
fail=0

echo "===== 闸门 0) 权重目录断言 + 服务自证（★防起错模型） ====="
SERVED=$(curl -s --max-time 10 "$API/models" || echo "")
echo "$SERVED" | tr ',' '\n' | grep '"id"' | sed 's/^/  /'
if ! echo "$SERVED" | grep -q "$API_MODEL"; then echo "FAIL: 服务未提供 $API_MODEL"; fail=1; fi
if ! echo "$SERVED" | grep -q "models/Qwen3-VL-32B-Instruct"; then
  echo "★FAIL: 服务 root 不是 bf16 目录（可能起成了 AWQ）"; fail=1
fi
echo "$SERVED" | grep -o "models/[A-Za-z0-9._-]*" | sort -u | sed 's/^/  root=/'

echo "===== 闸门 1) 管线与词表（自动探测位置） ====="
PIPE=""
for c in $R/_qwen_full_pipeline.py /root/_qwen_full_pipeline.py; do
  [ -f "$c" ] && { PIPE="$c"; break; }
done
[ -z "$PIPE" ] && { echo "FAIL: 找不到管线"; fail=1; }
echo "  管线 $PIPE"
CORES=""
for c in $R/_ordinal_cores_r2.json $R/_ordinal_cores_v3.json /root/_ordinal_cores_r2.json; do
  [ -f "$c" ] && { CORES="$c"; break; }
done
if [ -z "$CORES" ]; then echo "FAIL: 找不到序数核心词表"; fail=1;
else $PY -c "import json;print('  词表',len(json.load(open('$CORES'))),'条 ←','$CORES')"; fi

echo "===== 闸门 2) 数据齐备 ====="
nq=$($PY -c "import json;print(len(json.load(open('$DATA/queries/queries.json'))))" 2>/dev/null || echo 0)
nimg=$(ls "$DATA/Images/visible" 2>/dev/null | wc -l)
echo "  queries=$nq  visible 图=$nimg"
[ "$nq" -ge 5690 ] && [ "$nimg" -ge 1295 ] || { echo "FAIL: 数据不齐"; fail=1; }

echo "===== 闸门 3) 输出与磁盘 ====="
mkdir -p "$(dirname "$OUT")"
touch "$OUT" || { echo "FAIL: 输出不可写"; fail=1; }
df -h "$(dirname "$OUT")" | tail -1 | sed 's/^/  /'

echo "===== 闸门 4) 管线常量完好 ====="
for pat in "FULL_DIRECT_PROMPT" "ENUM_PROMPT" "SEL_PROMPT" "make_gen_api"; do
  n=$(grep -c "$pat" "$PIPE" 2>/dev/null || echo 0)
  echo "  $pat × $n"
  [ "$n" -ge 1 ] || { echo "FAIL: 缺 $pat"; fail=1; }
done

echo "===== 闸门 5) 接线自校验（串行同条件；必须逐位一致） ====="
SMOKE=""
for c in $R/_smoke_qwen32b.py /root/_smoke_qwen32b.py; do
  [ -f "$c" ] && { SMOKE="$c"; break; }
done
[ -z "$SMOKE" ] && { echo "FAIL: 找不到 _smoke_qwen32b.py"; fail=1; }
if [ -n "$SMOKE" ]; then
  QWEN_ENGINE=api32b QWEN_API_BASE="$API" QWEN_API_MODEL="$API_MODEL" \
  QWEN_DATA="$DATA" QWEN_CORES="$CORES" QWEN_CONC=1 \
    timeout 900 $PY -u "$SMOKE" --mode both --data "$DATA" 2>&1 | tail -12 | sed 's/^/  /'
  s=${PIPESTATUS[0]}
  echo "  烟测退出码 $s"
  [ "$s" -eq 0 ] || { echo "FAIL: 接线自校验未通过"; fail=1; }
fi

echo "===== 结论 ====="
if [ "$fail" -ne 0 ]; then echo "BLOCKED: 拒绝启动全量"; exit 2; fi
echo "PASS: 闸门全过"

if pgrep -f "_qwen_full_pipeline.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"; exit 0
fi

MINI=${QWEN_ONLY_FIRST:-}
cd $R
ENVS="QWEN_ENGINE=api32b QWEN_API_BASE=$API QWEN_API_MODEL=$API_MODEL QWEN_CONC=$CONC"
ENVS="$ENVS QWEN_PHASE=A QWEN_ONLY_NONORD=1 QWEN_DATA=$DATA QWEN_CORES=$CORES"
ENVS="$ENVS QWEN_OUT_DIRECT=$OUT QWEN_OUT_SOM=$SOMOUT"
eval "env $ENVS setsid nohup $PY -u $PIPE > $LOG 2>&1 < /dev/null &"
sleep 30
pid=$(pgrep -f _qwen_full_pipeline.py | head -1)
echo "已启动 PID=${pid:-无}"
echo "---- 日志首 22 行 ----"
head -n 22 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/  /'
echo "---- 输出文件 ----"
ls -la "$OUT" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="