echo "=========== GPU ==========="
nvidia-smi
echo "=========== 进程 (按内存 Top8) ==========="
ps aux --sort=-%mem | head -9
echo "=========== 内存 ==========="
free -h
echo "=========== 磁盘 ==========="
df -h / /root 2>/dev/null
echo "=========== 负载/运行时长 ==========="
uptime
echo "=========== 我们的推理进程 ==========="
pgrep -af 'predict_gd_3modal.py' || echo "已结束(无进程)"
