#!/bin/bash
# M 件第 2b 步：坐标口径修正为 whole 后**重跑 20 条冒烟**（重画 _sheet.png）
set -u
R=/root/rivermind-data

export QWEN_MODALITIES=3m QWEN_RULES5=1 QWEN_MULTIBOX=1 QWEN_MAXNEW=1024 \
       QWEN_THINKING=0 QWEN_MERGE_SPACE=whole \
       QWEN_ENGINE=api32b QWEN_API_MODEL=qwen38-27b \
       QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json

echo "=========== 冒烟配置（= 全量配置）==========="
echo "  MODALITIES=$QWEN_MODALITIES RULES5=$QWEN_RULES5 MULTIBOX=$QWEN_MULTIBOX"
echo "  MAXNEW=$QWEN_MAXNEW THINKING=$QWEN_THINKING MERGE_SPACE=$QWEN_MERGE_SPACE"
echo "==========================================="
echo

# 记住旧的（未换算）证据，便于对比
[ -f "$R/_q38_gate20.json" ] && cp -f "$R/_q38_gate20.json" "$R/_q38_gate20_panel.json"
[ -f "$R/q38_gate_vis/_sheet.png" ] && cp -f "$R/q38_gate_vis/_sheet.png" "$R/_sheet_panel.png"

bash "$R/_q38_gate.sh"
