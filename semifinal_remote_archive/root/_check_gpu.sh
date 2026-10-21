#!/bin/bash
# GPU 可用性专项排查（其余都已确认）
set -u
echo "=== nvidia-smi 完整输出 ==="
nvidia-smi 2>&1 | head -20 || echo "  退出码 $?"
echo; echo "=== which / 路径 ==="
command -v nvidia-smi || echo "  PATH 里没有 nvidia-smi"
for p in /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi; do
  [ -x "$p" ] && echo "  存在可执行: $p"
done
echo; echo "=== 设备节点 ==="
ls -l /dev/nvidia* 2>/dev/null | head -8 || echo "  ★无 /dev/nvidia* 设备节点"
echo; echo "=== 驱动版本 ==="
cat /proc/driver/nvidia/version 2>/dev/null || echo "  ★无 /proc/driver/nvidia/version"
echo; echo "=== 环境变量（NVIDIA_* / CUDA_*）==="
env | grep -Ei 'nvidia|cuda|visible_devices' | sed 's/^/  /' || echo "  (无)"
echo; echo "=== torch 是否看得到 CUDA ==="
/root/rivermind-data/venv/bin/python - <<'PYEOF' 2>&1 | tail -6
import torch
print("  torch:", torch.__version__)
print("  cuda available:", torch.cuda.is_available())
print("  device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("  device 0:", torch.cuda.get_device_name(0))
PYEOF
echo; echo "=== /data/tini 复查 ==="
ls -ld /data /data/tini 2>&1 | sed 's/^/  /'
df -h /data /data/tini 2>&1 | sed 's/^/  /'
echo; echo "=== 是否有别的 nvidia 挂载 ==="
mount 2>/dev/null | grep -i nvidia | sed 's/^/  /' || echo "  (无 nvidia 挂载)"
echo "=== 排查结束 ==="