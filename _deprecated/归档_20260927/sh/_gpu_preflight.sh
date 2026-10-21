#!/bin/bash
# 只读：跑批前 GPU 空闲窗口检查
set +e
echo "===== 时间 ====="; date '+%Y-%m-%d %H:%M:%S'
echo "===== GPU 显存/利用率 ====="
nvidia-smi --query-gpu=memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
echo "===== GPU 计算进程 ====="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "===== 是否还有 test_visualize_20b / 其他推理进程 ====="
ps -ef | grep -v grep | grep -E "test_visualize|locate_run|python.*\.py" | head
echo "===== myconda transformers（应仍 5.17，未被我们改动）/ overlay（4.52.4）====="
/root/miniconda3/envs/myconda/bin/python -c "import transformers;print('myconda',transformers.__version__)" 2>/dev/null
/root/locate_venv/bin/python -c "import transformers;print('overlay',transformers.__version__)" 2>/dev/null