echo "=== procs ==="
pgrep -af 'predict_gd_3modal.py' | grep -v pgrep | head
echo "=== run.log tail ==="
tail -12 /root/infer/run.log
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
