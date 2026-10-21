#!/usr/bin/env bash
# ============================================================
# 云端全量推理一键编排：前置检查 → 全量推理 → 导出提交 → 快照归档 → 打包
#
# 用法：
#   bash scripts/run_full.sh --config configs/cloud_gdino.yaml --split 复赛 --desc base
#   bash scripts/run_full.sh --config configs/cloud_qwen.yaml --split 复赛 --desc qwen_multi
#
# 产物：
#   outputs/<split>/predictions_<desc>.json
#   outputs/<split>/archive/run_<ts>_<desc>/  （predictions/submission/三快照/run.log/MD5SUMS）
#   outputs/<split>/archive/run_<ts>_<desc>.tar.gz
# ============================================================
set -euo pipefail

CONFIG="configs/cloud_qwen.yaml"
SPLIT="复赛"
DESC="run"
DATA_ROOT=""
SKIP_PREFLIGHT=""

# 解析 python：优先显式 --python / 环境变量，其次 PATH，再退到 conda
if [ -z "${PYTHON:-}" ]; then
  if command -v python >/dev/null 2>&1; then PYTHON=python
  elif [ -x /root/miniconda3/bin/python ]; then PYTHON=/root/miniconda3/bin/python
  elif [ -x /opt/conda/bin/python ]; then PYTHON=/opt/conda/bin/python
  else PYTHON=python3; fi
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --split) SPLIT="$2"; shift 2;;
    --desc) DESC="$2"; shift 2;;
    --data-root) DATA_ROOT="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift;;
    *) echo "未知参数：$1"; exit 2;;
  esac
done

DR=()
[ -n "$DATA_ROOT" ] && DR=(--data-root "$DATA_ROOT")

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/${SPLIT}"
RUN_NAME="run_${TS}_${DESC}"
RUN_DIR="${OUT_DIR}/archive/${RUN_NAME}"
PRED="${OUT_DIR}/predictions_${DESC}.json"
mkdir -p "$OUT_DIR" "$RUN_DIR"
LOG="${RUN_DIR}/run.log"

# 同时输出到终端与 run.log
exec > >(tee -a "$LOG") 2>&1

echo "================ 云端全量推理 ================"
echo "config=$CONFIG  split=$SPLIT  desc=$DESC"
echo "python=$PYTHON  run_dir=$RUN_DIR"
echo "time=$(date)"
echo "----------------------------------------------"

# 0) 环境快照（最先落，哪怕后面失败也已记录环境/代码）
$PYTHON scripts/snapshot_run.py --out-dir "$RUN_DIR" --config "$CONFIG" \
  --label "${SPLIT}-${DESC}" "${DR[@]}"

# 1) 前置检查
if [ -z "$SKIP_PREFLIGHT" ]; then
  echo "== pytest =="
  $PYTHON -m pytest tests/ -q
  echo "== 冒烟（前 2 图）=="
  $PYTHON scripts/smoke_test.py --config "$CONFIG" "${DR[@]}"
fi

# 2) 全量推理（计时；GNU time 不存在则直接跑）
echo "== 全量推理 → $PRED =="
if command -v /usr/bin/time >/dev/null 2>&1; then
  /usr/bin/time -v $PYTHON scripts/infer.py --config "$CONFIG" --output "$PRED" "${DR[@]}"
else
  $PYTHON scripts/infer.py --config "$CONFIG" --output "$PRED" "${DR[@]}"
fi

# 3) 导出提交包
echo "== 导出提交包 =="
$PYTHON scripts/export_submission.py --config "$CONFIG" --predictions "$PRED" "${DR[@]}"

# 4) 结果落位 run 目录 + MD5
cp -f "$PRED"                    "$RUN_DIR/predictions.json"
cp -f outputs/submission.json    "$RUN_DIR/submission.json"
cp -f outputs/submission.zip     "$RUN_DIR/submission.zip"

# 4b) 诊断明细回传（多变量同轮归因的唯一凭证）
# 没有它，事后无法离线还原 no_chain / no_depth_prior 臂，也无法分摊
# max_new_tokens 与 multi_box_id 的功劳 —— 即无法从一次运行里分离多个变量。
DIAG="outputs/diag/diag_raw_predictions_${DESC}.jsonl"
EXTRA_MD5=""
if [ -f "$DIAG" ]; then
  cp -f "$DIAG" "$RUN_DIR/diag_raw.jsonl"
  EXTRA_MD5="diag_raw.jsonl"
  echo "== 诊断明细已归档：$RUN_DIR/diag_raw.jsonl（$(wc -l < "$DIAG") 行）=="
else
  echo "!! 警告：未找到诊断明细 $DIAG"
  echo "!! 若 config 里 diag_dump=false，本轮将无法做多变量归因。"
fi
( cd "$RUN_DIR" && md5sum predictions.json submission.json submission.zip \
    $EXTRA_MD5 config_snapshot.yaml env_snapshot.json code_snapshot.json > MD5SUMS.txt )

# 5) 打包回传
tar -czf "${RUN_DIR}.tar.gz" -C "${OUT_DIR}/archive" "${RUN_NAME}"
echo "== 完成 =="
echo "  run 目录：$RUN_DIR"
echo "  回传包  ：${RUN_DIR}.tar.gz"
echo "  MD5     ："
cat "$RUN_DIR/MD5SUMS.txt"
