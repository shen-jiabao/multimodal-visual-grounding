#!/bin/bash
# 新服务器（gpuhome）环境侦察：为"最小环境清单"提供实测依据。
# 只读操作，不改任何东西。
echo "=== 主机 ==="
hostname; uname -sr; cat /etc/os-release 2>/dev/null | head -2
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
nvidia-smi | tail -6
echo
echo "=== 磁盘 ==="
df -h / /root 2>/dev/null | grep -v tmpfs
echo
echo "=== 内存 / CPU ==="
free -g | head -2; nproc
echo
echo "=== 关键目录 ==="
for d in /root/models /srv /srv/models /srv/dataset /root/assets /data /workspace /mnt; do
  [ -e "$d" ] && echo "  有 $d" || echo "  无 $d"
done
echo
echo "=== 是否自带 vLLM 服务（8000 端口）==="
ss -ltnp 2>/dev/null | grep -E ':8000|:5000' || echo "  8000/5000 未监听"
pgrep -af vllm | head -3 || echo "  无 vllm 进程"
echo
echo "=== python ==="
which python python3 2>/dev/null
python3 -V
echo
echo "=== 关键包（逐个查，避免一行式被引号破坏）==="
python3 - <<'PYEOF'
import importlib.util as u
for n in ("torch", "transformers", "vllm", "openai", "PIL", "numpy",
          "modelscope", "accelerate", "peft", "cv2"):
    s = u.find_spec(n)
    if not s:
        print("  " + n.ljust(13) + "MISSING")
        continue
    try:
        m = __import__(n)
        print("  " + n.ljust(13) + str(getattr(m, "__version__", "?")))
    except Exception as e:
        print("  " + n.ljust(13) + "装了但导入失败: " + type(e).__name__)
PYEOF
echo
echo "=== 模型目录内容（若存在）==="
ls -d /root/models/*/ 2>/dev/null || echo "  /root/models 为空或不存在"
echo
echo "=== 数据集是否已在场 ==="
if [ -f /srv/dataset/queries/queries.json ]; then
  echo "  queries 存在"
  ls /srv/dataset/Images/visible 2>/dev/null | wc -l
else
  echo "  /srv/dataset 不存在（需上传）"
fi