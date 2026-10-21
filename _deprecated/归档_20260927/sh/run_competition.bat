@echo off
setlocal
cd /d "d:\Desktop\2026AiC"
if not exist "checkpoints\competition" mkdir "checkpoints\competition"
python train_competition.py --image_size 224 --epochs 50 --batch_size 16 --lr 2e-4 --log_interval 100 --output_dir checkpoints/competition --image_backbone resnet10t > "checkpoints\competition\train_log.txt" 2>&1
echo.
echo === 训练完成 ===
echo 预测文件：result.json 和 result.zip
endlocal
