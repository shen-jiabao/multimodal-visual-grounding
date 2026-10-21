#!/bin/bash
# M 件第 2 步：闸门 G2/G3 冒烟（20 条），跑**与全量完全相同的 M 配置**
set -u
R=/root/rivermind-data

export QWEN_MODALITIES=3m QWEN_RULES5=1 QWEN_MULTIBOX=1 QWEN_MAXNEW=1024 \
       QWEN_THINKING=0 QWEN_ENGINE=api32b QWEN_API_MODEL=qwen38-27b \
       QWEN_DATA=$R/dataset QWEN_CORES=$R/_ordinal_cores_r2.json

echo "=========== 冒烟配置（= 全量配置）==========="
echo "  MODALITIES=$QWEN_MODALITIES RULES5=$QWEN_RULES5 MULTIBOX=$QWEN_MULTIBOX"
echo "  MAXNEW=$QWEN_MAXNEW THINKING=$QWEN_THINKING MODEL=$QWEN_API_MODEL"
echo "==========================================="
echo

bash "$R/_q38_gate.sh"
