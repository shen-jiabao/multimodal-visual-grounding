#!/usr/bin/env bash
# Linux/macOS 一键脚本：数据准备 -> 训练 -> 评估 -> 预测
# 离线模式（不下载预训练权重）

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="${PROJECT_ROOT}/基于大模型的多模态视觉理解与推理-示例数据"

echo "=== [1/4] 准备数据 ==="
python "${PROJECT_ROOT}/prepare_data.py" --data_root "${DATA_ROOT}" --val_ratio 0.1

echo ""
echo "=== [2/4] 训练（演示，2 个 epoch） ==="
python "${PROJECT_ROOT}/train.py" \
    --offline --image_backbone resnet10t --epochs 2 \
    --batch_size 4 --num_workers 0 \
    --output_dir "${PROJECT_ROOT}/checkpoints/exp"

echo ""
echo "=== [3/4] 评估 ==="
python "${PROJECT_ROOT}/evaluate.py" \
    --ckpt "${PROJECT_ROOT}/checkpoints/exp/best.pth" \
    --val_json val.json

echo ""
echo "=== [4/4] 预测 & 生成 zip ==="
python "${PROJECT_ROOT}/predict.py" \
    --ckpt "${PROJECT_ROOT}/checkpoints/exp/best.pth" \
    --test_json test.json \
    --output "${PROJECT_ROOT}/result.json" \
    --save_zip

echo ""
echo "全部完成！结果文件: result.json / result.zip"
