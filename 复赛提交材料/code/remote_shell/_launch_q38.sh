#!/bin/bash
# 全量启动器（换模型路线通用）：重跑 4587 条**非序数**直接定位
#   与 _launch_q32bf16.sh 的唯一差别（单变量前提：只换模型）：
#     · 服务模型名      → 由 QWEN_API_MODEL 注入（I 件 qwen38-27b／K 件 qwen25vl-72b-awq）
#     · 服务 root 断言  → 由 QWEN_ROOT_MUST / QWEN_ROOT_FORBID 注入（见下）
#     · 输出/日志三个路径 → 由 QWEN_OUT / QWEN_LOG / QWEN_OUT_SOM 注入
#   提示词 / 解码 / 并发**逐字不变**（CONC=8，与 bf16 那次相同）。
#
#   ★闸门 0 的 root 断言**必须按件注入**（默认值 = I 件的值，行为逐字不变）：
#     I 件：QWEN_ROOT_MUST=models/Qwen3.8-27B   QWEN_ROOT_FORBID=32B-Instruct
#     K 件：QWEN_ROOT_MUST=models/Qwen2.5-VL-72B  QWEN_ROOT_FORBID=Qwen3
#           （FORBID=Qwen3 一次拦住两个错模型：Qwen3.8-27B 与 Qwen3-VL-32B-Instruct；
#             而 Qwen2.5-VL-72B 不含 "Qwen3" ⇒ 不会被误拦。）
#     ★若照抄 I 件的硬编码，K 件会在闸门 0 被**假报失败**（断言的是别的模型）。
#
#   ★闸门 5 的替换说明（本件特有，非放松）：
#     bf16 那次的闸门 5 是"我方管线 vs 归档 Api32B 逐位一致"——它成立的前提是
#     **同一个模型**。换模型的件与 32B 归档实现比对**必然不一致**，该判据
#     在本件上无意义（若照抄会假报失败）。故替换为「G2/G3 冒烟证据在场」：
#     读 `_q38_gate20.json`，断言可解析 ≥16（= 预登记 I §4 的 G2 判据）。
#     G3 是人眼判据，由用户判定，证据在 q38_gate_vis/。
set -u
R=/root/rivermind-data
PY=${QWEN_PY:-$R/venv/bin/python}
DATA=${QWEN_DATA:-$R/dataset}
OUT=${QWEN_OUT:-$R/output/qwen38_direct.json}
LOG=${QWEN_LOG:-$R/qwen38_run.log}
SOMOUT=${QWEN_OUT_SOM:-$R/output/qwen38_som.json}
API=${QWEN_API_BASE:-http://127.0.0.1:8000/v1}
API_MODEL=${QWEN_API_MODEL:-qwen38-27b}
ROOT_MUST=${QWEN_ROOT_MUST:-models/Qwen3.8-27B}
ROOT_FORBID=${QWEN_ROOT_FORBID:-32B-Instruct}
SPACE=${QWEN_BBOX_SPACE:-}
QIDS=${QWEN_QIDS:-}
CONC=${QWEN_CONC:-8}
GATE=${QWEN_GATE:-$R/_q38_gate20.json}
fail=0

echo "===== 闸门 0) 服务自证（★防起错模型） ====="
SERVED=$(curl -s --max-time 10 "$API/models" || echo "")
echo "$SERVED" | tr ',' '\n' | grep '"id"' | sed 's/^/  /'
if ! echo "$SERVED" | grep -q "$API_MODEL"; then echo "FAIL: 服务未提供 $API_MODEL"; fail=1; fi
if ! echo "$SERVED" | grep -q "$ROOT_MUST"; then
  echo "★FAIL: 服务 root 不是 $ROOT_MUST"; fail=1
fi
if echo "$SERVED" | grep -q "$ROOT_FORBID"; then
  echo "★FAIL: 服务 root 出现 $ROOT_FORBID（起错模型）"; fail=1
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

echo "===== 闸门 5) G2/G3 冒烟证据（★本件替换项，见文件头说明） ====="
if [ ! -f "$GATE" ]; then
  echo "FAIL: 缺 $GATE（G2 冒烟未跑）"; fail=1
else
  $PY - "$GATE" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = [k for k, v in d.items() if v.get("bbox")]
print(f"  冒烟 {len(d)} 条｜可解析 {len(ok)}/{len(d)}（要求 ≥16）")
sys.exit(0 if len(ok) >= 16 and len(d) >= 20 else 1)
PY
  [ $? -eq 0 ] || { echo "FAIL: G2 未达 ≥16/20"; fail=1; }
fi

echo "===== 结论 ====="
if [ "$fail" -ne 0 ]; then echo "BLOCKED: 拒绝启动全量"; exit 2; fi
echo "PASS: 闸门全过"

if pgrep -f "_qwen_full_pipeline.py" > /dev/null; then
  echo "已在运行 PID=$(pgrep -f _qwen_full_pipeline.py | head -1)"; exit 0
fi

cd $R
ENVS="QWEN_ENGINE=api32b QWEN_API_BASE=$API QWEN_API_MODEL=$API_MODEL QWEN_CONC=$CONC"
ENVS="$ENVS QWEN_PHASE=A QWEN_ONLY_NONORD=1 QWEN_DATA=$DATA QWEN_CORES=$CORES"
ENVS="$ENVS QWEN_OUT_DIRECT=$OUT QWEN_OUT_SOM=$SOMOUT"
# 坐标空间适配（Qwen2.5-VL 需 smart_resize_px；缺省不注入 ⇒ 既有件行为逐字不变）
if [ -n "$SPACE" ]; then ENVS="$ENVS QWEN_BBOX_SPACE=$SPACE"; fi
# 作用范围收窄（预登记 L：只在 429 个位置跑；缺省不注入 ⇒ 全量）
if [ -n "$QIDS" ]; then
  ENVS="$ENVS QWEN_QIDS=$QIDS"
  echo "  作用范围收窄：$QIDS（$($PY -c "import json;print(len(json.load(open('$QIDS'))))" 2>/dev/null) 条）"
fi
eval "env $ENVS setsid nohup $PY -u $PIPE > $LOG 2>&1 < /dev/null &"
sleep 30
pid=$(pgrep -f _qwen_full_pipeline.py | head -1)
echo "已启动 PID=${pid:-无}"
echo "---- 日志首 22 行 ----"
head -n 22 "$LOG" 2>/dev/null | cut -c1-170 | sed 's/^/  /'
echo "---- 输出文件 ----"
ls -la "$OUT" 2>/dev/null | sed 's/^/  /'
echo "=== 结束 ==="
