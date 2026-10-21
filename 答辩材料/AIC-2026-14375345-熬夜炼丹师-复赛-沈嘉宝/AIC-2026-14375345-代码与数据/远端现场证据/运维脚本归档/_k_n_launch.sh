#!/bin/bash
# 预登记 N 启动：只在 519 个位置（q38 改动集 = VLM 框所在位）跑 K
set -u
R=/root/rivermind-data
mkdir -p "$R/output"
echo "作用域清单：$R/k_scope519.json"
env \
  QWEN_API_MODEL=qwen25vl-72b-awq \
  QWEN_ROOT_MUST=models/Qwen2.5-VL-72B \
  QWEN_ROOT_FORBID=Qwen3 \
  QWEN_BBOX_SPACE=smart_resize_px \
  QWEN_QIDS="$R/k_scope519.json" \
  QWEN_OUT="$R/output/k72b_direct_519.json" \
  QWEN_OUT_SOM="$R/output/k72b_som_519.json" \
  QWEN_LOG="$R/k72b_n.log" \
  QWEN_GATE="$R/_k_gate20.json" \
  bash "$R/_launch_q38.sh"
