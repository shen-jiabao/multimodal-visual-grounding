pgrep -af 'predict_gd_3modal.py' || echo "NO_PROC"
echo "=== run.log size ==="
ls -l /root/infer/run.log 2>&1
echo "=== tail ==="
tail -15 /root/infer/run.log 2>&1
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
