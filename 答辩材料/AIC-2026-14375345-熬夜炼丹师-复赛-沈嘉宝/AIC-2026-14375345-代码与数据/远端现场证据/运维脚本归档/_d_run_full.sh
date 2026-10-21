#!/bin/bash
# D 全量（作用范围 1652 条）——**分离式启动**：立即返回，不占 SSH 通道
#
# ⚠ 本次运行**违反 `_prereg_D.md` §8**（小样判据 ① ② 均未过 ⇒ 不做全量）。
#   这是用户在看到"判据未过 / 上界 0.7805 / 大概率为负"之后作出的**显式覆盖**，
#   已留痕于 `_frozen_criteria__跑后实录.md` §十二。
set -u
R=/root/rivermind-data
OUT=$R/output/qwen32b_d3_full.json
OUTS=$R/output/qwen32b_d3_full_som.json
LOG=$R/d_full_run.log

if pgrep -f 'qwen_full_pipeline' >/dev/null; then
  echo "★ 已有管线在跑，拒绝重复启动："
  pgrep -af 'qwen_full_pipeline' | grep -v 'bash -c'
  exit 1
fi

mkdir -p "$R/output"
rm -f "$OUT" "$OUTS"          # 全新一次全量，不与小样产物混
cd "$R"

echo "=== 分离式启动 D 全量（作用范围 1652 条 / 707 张图）==="
QWEN_ENGINE=api32b \
QWEN_MODALITIES=3 \
QWEN_PHASE=A \
QWEN_ONLY_NONORD=1 \
QWEN_QIDS=$R/_d_scope.json \
QWEN_DATA=$R/dataset \
QWEN_CORES=$R/_ordinal_cores_v3.json \
QWEN_OUT_DIRECT="$OUT" \
QWEN_OUT_SOM="$OUTS" \
QWEN_CONC=8 \
setsid nohup "$R/venv/bin/python" -u "$R/_qwen_full_pipeline.py" > "$LOG" 2>&1 &
PID=$!
echo "  已分离启动（setsid nohup），shell 返回的 PID=$PID"
sleep 30
echo "  实际管线 PID: $(pgrep -f '_qwen_full_pipeline.py' | head -1)"
echo "  日志尾 3 行:"
tail -3 "$LOG"
echo "  待处理图: $(grep '待处理图' "$LOG" | tail -1)"
echo "  （本脚本立即返回；进度用 `tail -3 $LOG` 查）"
echo "=== 启动结束 ==="