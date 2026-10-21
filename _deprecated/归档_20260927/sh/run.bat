@echo off
REM =====================================================================
REM  多模态视觉定位 —— 竞赛标准训练脚本（离线模式）
REM  Multi-Modal Visual Grounding —— Competition Training Script (Offline)
REM =====================================================================
chcp 65001 > nul
setlocal

REM ---- 设置离线环境变量 ----
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set TORCH_HOME=%USERPROFILE%\.cache\torch

set DATA_ROOT=%~dp0基于大模型的多模态视觉理解与推理-示例数据
set OUTPUT_DIR=%~dp0checkpoints\competition

echo.
echo ============================================================
echo   多模态视觉定位 - 竞赛标准训练（离线模式）
echo   图像编码器: resnet50 (随机初始化)
echo   文本编码器: LSTM (随机初始化，无预训练 BERT)
echo   融合层: 4层 Transformer Decoder, 8头, dim=1024
echo   图像尺寸: 448x448
echo   训练轮次: 80 epochs
echo   数据增强: 随机水平翻转 + 颜色抖动
echo ============================================================
echo.

REM ---- 步骤 1: 数据准备 ----
echo [1/4] 准备数据（train/val/test）...
if not exist "%DATA_ROOT%\train.json" (
    copy "%DATA_ROOT%\sample.json" "%DATA_ROOT%\train.json" /Y
)
if not exist "%DATA_ROOT%\val.json" (
    copy "%DATA_ROOT%\sample.json" "%DATA_ROOT%\val.json" /Y
)
if not exist "%DATA_ROOT%\test.json" (
    copy "%DATA_ROOT%\sample.json" "%DATA_ROOT%\test.json" /Y
)
python "%~dp0prepare_data.py" --data_root "%DATA_ROOT%" --no_vis

echo.
REM ---- 步骤 2: 训练（离线模式：--offline） ----
echo [2/4] 开始训练（离线配置：resnet50 + LSTM文本编码器）...
python "%~dp0train.py" ^
    --offline ^
    --data_root "%DATA_ROOT%" ^
    --image_backbone resnet50 ^
    --text_encoder distilbert-base-uncased ^
    --epochs 80 ^
    --batch_size 8 ^
    --lr 1e-3 ^
    --weight_decay 1e-4 ^
    --warmup_epochs 5 ^
    --grad_clip 1.0 ^
    --lambda_l1 1.0 --lambda_iou 3.0 --lambda_giou 2.0 ^
    --num_workers 0 ^
    --log_interval 10 ^
    --output_dir "%OUTPUT_DIR%"

if errorlevel 1 (
    echo.
    echo [ERROR] 训练失败，请检查日志
    endlocal
    exit /b 1
)

echo.
REM ---- 步骤 3: 验证评估 ----
echo [3/4] 模型评估（在 val.json 上）...
python "%~dp0evaluate.py" ^
    --ckpt "%OUTPUT_DIR%\best.pth" ^
    --val_json val.json

echo.
REM ---- 步骤 4: 预测并生成提交文件 ----
echo [4/4] 预测测试集并生成 result.json 和 result.zip...
python "%~dp0predict.py" ^
    --ckpt "%OUTPUT_DIR%\best.pth" ^
    --test_json test.json ^
    --output "%~dp0result.json" ^
    --save_zip

echo.
echo ============================================================
echo   全部完成！
echo   - 最优模型: %OUTPUT_DIR%\best.pth
echo   - 预测结果: %~dp0result.json
echo   - 提交文件: %~dp0result.zip
echo ============================================================

endlocal
