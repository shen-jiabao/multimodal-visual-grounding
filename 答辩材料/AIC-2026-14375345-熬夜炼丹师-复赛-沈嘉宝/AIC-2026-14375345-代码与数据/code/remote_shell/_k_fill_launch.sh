#!/bin/bash
# 预登记 L（B′ 补漏版）启动：只在 429 个位置跑 K
set -u
R=/root/rivermind-data
mkdir -p "$R/output"
echo "作用域清单：$R/k_scope429.json"
env \
  QWEN_API_MODEL=qwen25vl-72b-awq \
  QWEN_ROOT_MUST=models/Qwen2.5-VL-72B \
  QWEN_ROOT_FORBID=Qwen3 \
  QWEN_BBOX_SPACE=smart_resize_px \
  QWEN_QIDS="$R/k_scope429.json" \
  QWEN_OUT="$R/output/k72b_direct_429.json" \
  QWEN_OUT_SOM="$R/output/k72b_som_429.json" \
  QWEN_LOG="$R/k72b_fill.log" \
  QWEN_GATE="$R/_k_gate20.json" \
  bash "$R/_launch_q38.sh"
